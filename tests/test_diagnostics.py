"""Self-serve diagnostics: launchd state parsing, `usagedash doctor`, and the install-time readiness wait.

Nothing here talks to this machine's launchd or to a listener it did not start: launchd and the port
holder are faked, and the health probe runs against sockets opened inside the test.
"""

import logging
import os
import socket
import stat
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.fakes.fake_run import FakeRun
from usagedash import agent, cli, doctor
from usagedash.config import Binaries, Settings, load_settings
from usagedash.doctor import Finding, HealthKind, HealthResult, Level, PortHolder
from usagedash.errors import ConfigError, UsageDashError

TAB = "\t"
PORT = 9123


def launchctl_listing(
    *, state: str = "running", pid: int | None = 838, runs: int = 1, last_exit: str = "(never exited)", signal: str = ""
) -> str:
    """A listing in the shape ``launchctl print`` writes, nested blocks and look-alike names included."""
    rows = [
        "gui/501/com.persona.ai-usage-dashboard = {",
        f"{TAB}active count = 1",
        f"{TAB}state = {state}",
        "",
        f"{TAB}environment = {{",
        f"{TAB}{TAB}HOME => /Users/persona",
        f"{TAB}}}",
        "",
        f"{TAB}runs = {runs}",
    ]
    if pid is not None:
        rows.append(f"{TAB}pid = {pid}")
    rows.append(f"{TAB}last exit code = {last_exit}")
    if signal:
        rows.append(f"{TAB}last terminating signal = {signal}")
    rows += [
        "",
        f"{TAB}resource coalition = {{",
        f"{TAB}{TAB}state = active",
        f"{TAB}}}",
        f"{TAB}job state = running",
        "}",
    ]
    return "\n".join(rows) + "\n"


class UsageDashErrorOfAnotherKind(UsageDashError):
    """An error that is neither a ConfigError nor an AgentError, to show the remedy does not depend on those two."""


RUNNING = agent.ServiceState("running", 838, 1, "(never exited)", None)
CRASHED = agent.ServiceState("spawn scheduled", None, 4, "1", None)


class TestParsingWhatLaunchdSays:
    def test_reads_a_running_agent(self) -> None:
        assert agent.parse_service_state(launchctl_listing()) == RUNNING

    def test_reads_an_agent_that_exited_and_waits_to_start_again(self) -> None:
        parsed = agent.parse_service_state(launchctl_listing(state="spawn scheduled", pid=None, runs=4, last_exit="1"))

        assert parsed == CRASHED
        assert parsed.last_run_failed

    def test_a_nested_blocks_state_never_replaces_the_services_own(self) -> None:
        parsed = agent.parse_service_state(launchctl_listing(state="waiting", pid=None))

        assert parsed.state == "waiting"

    def test_only_the_first_top_level_occurrence_of_a_field_counts(self) -> None:
        parsed = agent.parse_service_state(f"{TAB}runs = 2\n{TAB}runs = 9\n")

        assert parsed.runs == 2

    def test_an_empty_listing_states_nothing(self) -> None:
        assert agent.parse_service_state("") == agent.ServiceState(None, None, None, None, None)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("838", 838), ("0", 0), ("-5", -5), ("", None), ("-", None), ("12abc", None), ("١٢", None), ("1.5", None)],
    )
    def test_a_number_is_read_only_when_it_is_plainly_one(self, text: str, expected: int | None) -> None:
        assert agent.parse_service_state(f"{TAB}pid = {text}\n").pid == expected

    @pytest.mark.parametrize(
        ("last_exit", "signal", "described"),
        [
            ("1", None, "exit code 1"),
            ("0", "(none)", "exit code 0"),
            ("(never exited)", None, "exit code (never exited)"),
            ("(never exited)", "Killed: 9", "signal Killed: 9"),
            ("1", "Segmentation fault: 11", "signal Segmentation fault: 11"),
        ],
    )
    def test_how_the_last_run_ended_names_the_signal_when_there_was_one(
        self, last_exit: str, signal: str | None, described: str
    ) -> None:
        assert agent.ServiceState(None, None, None, last_exit, signal).last_exit_description == described

    def test_a_field_longer_than_the_limit_is_cut(self) -> None:
        parsed = agent.parse_service_state(f"{TAB}state = {'x' * 500}\n")

        assert parsed.state == "x" * agent.MAX_SERVICE_FIELD_LENGTH

    @pytest.mark.parametrize(
        ("last_exit", "signal", "failed"),
        [
            ("(never exited)", None, False),
            ("0", None, False),
            ("1", None, True),
            ("78", None, True),
            ("-1", None, True),
            ("not a number", None, False),
            (None, None, False),
            ("(never exited)", "Killed: 9", True),
            ("0", "(none)", False),
            ("0", "(never signalled)", False),
            ("0", "", False),
        ],
    )
    def test_a_run_failed_only_on_a_nonzero_exit_or_a_signal(
        self, last_exit: str | None, signal: str | None, failed: bool
    ) -> None:
        assert agent.ServiceState(None, None, None, last_exit, signal).last_run_failed is failed


class TestAskingLaunchd:
    @pytest.fixture
    def paths(self, tmp_path: Path) -> agent.AgentPaths:
        return agent.agent_paths(home=tmp_path, user="persona")

    def test_service_state_parses_the_listing_launchd_prints(
        self, paths: agent.AgentPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout=launchctl_listing())
        monkeypatch.setattr(agent.subprocess, "run", run)

        assert agent.service_state(paths) == RUNNING
        assert run.commands == [["/bin/launchctl", "print", paths.service_target]]

    def test_service_state_is_none_when_launchd_has_no_such_service(
        self, paths: agent.AgentPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun()
        run.returncodes = {"print": agent.LAUNCHCTL_NO_SUCH_SERVICE}
        monkeypatch.setattr(agent.subprocess, "run", run)

        assert agent.service_state(paths) is None
        assert agent.is_loaded(paths) is False

    def test_any_other_launchctl_failure_is_an_error_not_an_absence(
        self, paths: agent.AgentPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "run", FakeRun(returncode=1, stderr="boom"))

        with pytest.raises(agent.AgentError, match=r"launchctl print failed \(1\): boom"):
            agent.service_state(paths)

    def test_a_service_with_an_empty_listing_is_still_loaded(
        self, paths: agent.AgentPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "run", FakeRun(stdout=""))

        assert agent.is_loaded(paths) is True
        assert agent.service_state(paths) == agent.ServiceState(None, None, None, None, None)


class TestPathOmissions:
    @staticmethod
    def executable(directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("#!/bin/sh\n")
        (directory / name).chmod(0o755)

    def test_an_unsafe_directory_holding_a_vendor_cli_is_flagged_and_one_that_is_not_is_not(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        shared, elsewhere = tmp_path / "shared", tmp_path / "not-there"
        self.executable(shared, "codex")
        shared.chmod(0o777)
        try:
            with caplog.at_level("WARNING", logger="usagedash.agent"):
                omissions = agent.agent_path_omissions(["codex"], f"{shared}:{elsewhere}")
        finally:
            shared.chmod(0o700)

        by_directory = {omission.directory: omission for omission in omissions}
        assert by_directory[str(shared)].holds_vendor_cli is True
        assert by_directory[str(elsewhere)].holds_vendor_cli is False
        assert caplog.messages == []

    def test_a_directory_that_passes_is_not_an_omission(self, tmp_path: Path) -> None:
        kept = tmp_path / "kept"
        self.executable(kept, "claude")

        omitted = [omission.directory for omission in agent.agent_path_omissions(["claude"], str(kept))]

        assert str(kept) not in omitted


@dataclass
class Machine:
    """What launchd, the port, and the network say, for the checks that ask them."""

    environment: dict[str, str] | None = field(default_factory=lambda: {"USAGEDASH_PORT": str(PORT)})
    state: agent.ServiceState | None = RUNNING
    health: HealthResult = field(default_factory=lambda: HealthResult(HealthKind.ANSWERED, 200))
    holder: PortHolder | None = None
    probed: list[str] = field(default_factory=list)


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Machine:
    fake = Machine()

    def probe(url: str) -> HealthResult:
        fake.probed.append(url)
        return fake.health

    monkeypatch.setattr(agent, "installed_environment", lambda _paths: fake.environment)
    monkeypatch.setattr(agent, "service_state", lambda _paths: fake.state)
    monkeypatch.setattr(doctor, "probe_health", probe)
    monkeypatch.setattr(doctor, "port_holder", lambda _port: fake.holder)
    return fake


@pytest.fixture
def paths(tmp_path: Path) -> agent.AgentPaths:
    return agent.agent_paths(home=tmp_path, user="persona")


def settings_for(tmp_path: Path, port: int = PORT) -> Settings:
    return load_settings(
        {
            "USAGEDASH_DATA_DIR": str(tmp_path / "data"),
            "USAGEDASH_PORT": str(port),
            "USAGEDASH_CLAUDE_BINARY": "claude",
            "USAGEDASH_CODEX_BINARY": "codex",
            "USAGEDASH_GROK_BINARY": "grok",
        },
        home=tmp_path,
    )


def _settings_outcome_for(tmp_path: Path) -> Settings | ConfigError:
    try:
        return settings_for(tmp_path)
    except ConfigError as error:
        return error


def titles(findings: list[Finding], level: Level | None = None) -> list[str]:
    return [finding.title for finding in findings if level is None or finding.level is level]


def only(findings: list[Finding], containing: str) -> Finding:
    matches = [finding for finding in findings if containing in finding.title]
    assert len(matches) == 1, titles(findings)
    return matches[0]


def executable_file(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(mode)
    return path


def write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestTheAgentChecks:
    def test_an_agent_that_was_never_installed_is_a_failure_that_names_the_install_command(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment, machine.state = None, None

        finding = only(doctor.diagnose_agent(paths, None), "not installed")

        assert finding.level is Level.FAIL
        assert finding.fix is not None
        assert "usagedash agent install" in finding.fix

    def test_an_agent_with_a_property_list_but_no_launchd_service_is_a_failure(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = None

        finding = only(doctor.diagnose_agent(paths, None), "has not loaded it")

        assert finding.level is Level.FAIL
        assert str(paths.plist) in finding.details[0]

    def test_a_service_whose_property_list_is_gone_is_a_warning_beside_the_running_finding(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment = None

        findings = doctor.diagnose_agent(paths, None)

        assert only(findings, "property list is gone").level is Level.WARN
        assert only(findings, "is running the agent").level is Level.OK

    def test_a_running_agent_is_reported_with_its_pid_and_launchd_facts(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        finding = only(doctor.diagnose_agent(paths, None), "is running the agent")

        assert finding.title == "launchd is running the agent (pid 838)"
        assert finding.details == ("state = running", "runs = 1", "last exit = (never exited)")

    def test_an_agent_that_crashed_is_a_failure_with_the_exit_code_and_the_log_lines(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = CRASHED
        write_log(paths.launchd_logs[1], "Traceback (most recent call last):\nOSError: [Errno 48] address in use\n")

        findings = doctor.diagnose_agent(paths, None)

        failure = only(findings, "started and stopped")
        assert failure.title == "The agent started and stopped (exit code 1)"
        assert f"after {agent.RESTART_THROTTLE_SECONDS} seconds" in failure.details[-1]
        assert only(findings, "launchd-stderr.log").details == (
            "Traceback (most recent call last):",
            "OSError: [Errno 48] address in use",
        )

    def test_an_agent_killed_by_a_signal_is_reported_by_the_signal(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = agent.ServiceState("waiting", None, 2, "(never exited)", "Killed: 9")

        assert only(doctor.diagnose_agent(paths, None), "started and stopped").title == (
            "The agent started and stopped (signal Killed: 9)"
        )

    def test_a_listed_agent_that_is_not_running_and_has_not_failed_is_a_warning(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = agent.ServiceState(None, None, None, None, None)

        assert only(doctor.diagnose_agent(paths, None), "not running it").level is Level.WARN

    def test_a_property_list_that_cannot_be_trusted_is_a_failure(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def untrusted(_paths: agent.AgentPaths) -> dict[str, str] | None:
            raise agent.AgentError("another account could change it")

        monkeypatch.setattr(agent, "installed_environment", untrusted)

        findings = doctor.diagnose_agent(paths, None)

        assert titles(findings, Level.FAIL) == [
            "The installed property list cannot be trusted: another account could change it"
        ]

    @staticmethod
    def untrusted_property_list(monkeypatch: pytest.MonkeyPatch) -> None:
        def untrusted(_paths: agent.AgentPaths) -> dict[str, str] | None:
            raise agent.AgentError("another account could change it")

        monkeypatch.setattr(agent, "installed_environment", untrusted)

    def test_an_untrusted_property_list_does_not_hide_a_running_agent(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.untrusted_property_list(monkeypatch)

        findings = doctor.diagnose_agent(paths, None)

        assert titles(findings) == [
            "The installed property list cannot be trusted: another account could change it",
            "launchd is running the agent (pid 838)",
        ]
        assert "property list is gone" not in " ".join(titles(findings))

    def test_an_untrusted_property_list_does_not_hide_a_crashed_agent_and_its_exit_status(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.untrusted_property_list(monkeypatch)
        machine.state = CRASHED

        findings = doctor.diagnose_agent(paths, None)

        assert titles(findings, Level.FAIL) == [
            "The installed property list cannot be trusted: another account could change it",
            "The agent started and stopped (exit code 1)",
        ]

    def test_an_untrusted_property_list_with_nothing_loaded_is_reported_once_and_not_as_not_installed(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.untrusted_property_list(monkeypatch)
        machine.state = None

        assert titles(doctor.diagnose_agent(paths, None)) == [
            "The installed property list cannot be trusted: another account could change it"
        ]

    def test_an_untrusted_property_list_and_an_unreachable_launchd_are_both_reported(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.untrusted_property_list(monkeypatch)

        def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

        monkeypatch.setattr(agent, "service_state", unreachable)

        assert titles(doctor.diagnose_agent(paths, None), Level.FAIL) == [
            "The installed property list cannot be trusted: another account could change it",
            "launchd cannot be asked about the agent: launchctl print could not be run: FileNotFoundError",
        ]

    def test_an_untrusted_property_list_names_uninstall_as_the_way_out_because_install_stops_at_the_same_error(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.untrusted_property_list(monkeypatch)

        finding = only(doctor.diagnose_agent(paths, None), "cannot be trusted")

        assert finding.fix is not None
        assert "`usagedash agent uninstall`" in finding.fix
        assert "removes a property list it cannot trust without starting it" in finding.fix
        assert "When the file is unsafe and not just unreadable, `agent install` alone reads it" in finding.fix
        assert "through the same trust check and stops at the same error" in finding.fix

    def test_installed_settings_that_cannot_be_read_because_the_property_list_cannot_are_cleared_with_uninstall(
        self,
    ) -> None:
        findings = doctor.settings_findings(None, agent.AgentError("cannot use the plist: another account can write"))

        assert findings[0].fix == doctor._RECOVER_THE_PROPERTY_LIST
        assert "`usagedash agent uninstall`" in findings[0].fix

    @pytest.mark.parametrize("error", [ConfigError("port must be an integer, got 'x'"), UsageDashErrorOfAnotherKind()])
    def test_installed_settings_that_are_invalid_are_corrected_and_installed_again_not_uninstalled(
        self, error: Exception
    ) -> None:
        assert isinstance(error, UsageDashError)

        findings = doctor.settings_findings(None, error)

        assert findings[0].fix == doctor._CORRECT_THE_SETTING
        assert "uninstall" not in findings[0].fix

    def test_launchd_that_cannot_be_asked_names_what_to_check(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

        monkeypatch.setattr(agent, "service_state", unreachable)

        finding = only(doctor.diagnose_agent(paths, None), "launchd cannot be asked")

        assert finding.fix == "Check that `/bin/launchctl` runs in this terminal, then run `usagedash doctor` again."

    def test_launchd_that_cannot_be_asked_is_a_failure(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

        monkeypatch.setattr(agent, "service_state", unreachable)

        assert titles(doctor.diagnose_agent(paths, None), Level.FAIL) == [
            "launchd cannot be asked about the agent: launchctl print could not be run: FileNotFoundError"
        ]


class TestSuggestingAPort:
    @staticmethod
    def free_except(*taken: int) -> Callable[[int], bool]:
        return lambda port: port not in taken

    def test_the_next_port_up_is_suggested_when_it_is_free(self) -> None:
        assert doctor.suggest_port(8787, self.free_except()) == 8788

    def test_taken_ports_are_skipped(self) -> None:
        assert doctor.suggest_port(8787, self.free_except(8788, 8789)) == 8790

    def test_at_the_highest_port_the_suggestion_comes_from_below_and_is_in_range(self) -> None:
        suggested = doctor.suggest_port(65535, self.free_except())

        assert suggested == 65534
        assert doctor.MIN_PORT <= suggested <= doctor.MAX_PORT

    def test_at_the_lowest_port_the_suggestion_comes_from_above(self) -> None:
        assert doctor.suggest_port(1024, self.free_except()) == 1025

    def test_ports_above_are_tried_before_ports_below_and_only_within_the_span(self) -> None:
        above = range(8788, 8787 + doctor.PORT_SUGGESTION_SPAN + 1)

        assert doctor.suggest_port(8787, self.free_except(*above)) == 8786
        assert doctor.suggest_port(8787, lambda port: port == 8787 + doctor.PORT_SUGGESTION_SPAN + 1) is None

    def test_no_port_is_suggested_when_none_nearby_is_free(self) -> None:
        assert doctor.suggest_port(8787, lambda _port: False) is None

    def test_the_suggestion_never_leaves_the_range_the_settings_accept(self) -> None:
        offered: list[int] = []

        def record(port: int) -> bool:
            offered.append(port)
            return False

        for port in (doctor.MIN_PORT, doctor.MAX_PORT):
            offered.clear()
            doctor.suggest_port(port, record)
            assert offered
            assert all(doctor.MIN_PORT <= candidate <= doctor.MAX_PORT for candidate in offered)
            assert port not in offered

    def test_a_port_someone_listens_on_is_not_free_and_a_released_one_is(self) -> None:
        with socket.socket() as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen()
            port = holder.getsockname()[1]

            assert doctor._port_is_free(port) is False

        assert doctor._port_is_free(port) is True


class TestWhetherTheDashboardAnswers:
    @pytest.fixture(autouse=True)
    def every_nearby_port_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "_port_is_free", lambda _port: True)
        monkeypatch.delenv("USAGEDASH_PORT", raising=False)

    def test_an_answering_dashboard_held_by_the_agent_is_fine(self, paths: agent.AgentPaths, machine: Machine) -> None:
        machine.holder = PortHolder(838, "python3.13")

        findings = doctor.diagnose_agent(paths, PORT)

        assert only(findings, "answers on port").level is Level.OK
        assert not [finding for finding in findings if finding.level in (Level.WARN, Level.FAIL)]
        assert machine.probed == [f"http://127.0.0.1:{PORT}/healthz"]

    def test_an_answer_from_a_process_that_is_not_the_agent_is_a_failure_that_names_it(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.holder = PortHolder(99, "python3.13")

        finding = only(doctor.diagnose_agent(paths, PORT), "is not the one holding it")

        assert finding.level is Level.FAIL
        assert (
            finding.title == f"The dashboard answers on port {PORT}, but the agent (pid 838) is not the one holding it"
        )
        assert "python3.13 (pid 99)" in finding.details[0]
        assert "usagedash serve" in finding.details[1]
        assert finding.fix == "Quit that process, then run `usagedash agent install`."

    @pytest.mark.parametrize(
        "state",
        [None, agent.ServiceState(None, None, None, None, None), CRASHED],
        ids=["not loaded", "loaded but not running", "crashed"],
    )
    def test_a_healthy_answer_is_not_ok_while_the_agent_is_not_running_and_a_visible_process_serves_it(
        self, paths: agent.AgentPaths, machine: Machine, state: agent.ServiceState | None
    ) -> None:
        machine.state, machine.holder = state, PortHolder(99, "python3.13")

        findings = doctor.diagnose_agent(paths, PORT)

        finding = only(findings, "the agent is not running")
        assert finding.level is Level.FAIL
        assert finding.title == (
            f"The dashboard answers on port {PORT}, but the agent is not running, so something else is serving it"
        )
        assert finding.details[0] == f"Port {PORT} is held by python3.13 (pid 99)."
        assert not [title for title in titles(findings) if title == f"The dashboard answers on port {PORT}"]

    def test_a_healthy_answer_is_not_ok_while_the_agent_is_not_running_even_when_the_holder_cannot_be_seen(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state, machine.holder = agent.ServiceState(None, None, None, None, None), None

        finding = only(doctor.diagnose_agent(paths, PORT), "the agent is not running")

        assert finding.level is Level.FAIL
        assert not [detail for detail in finding.details if detail.startswith("Port ")]

    def test_the_report_does_not_end_in_working_when_the_agent_is_not_the_one_serving(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state, machine.holder = agent.ServiceState(None, None, None, None, None), PortHolder(99, "python3.13")

        findings = doctor.diagnose_agent(paths, PORT)
        report = doctor.render(findings, tmp_path)

        assert doctor.has_failure(findings) is True
        assert "Result: working" not in report
        assert "Result: 1 problem found." in report

    def test_another_program_answering_with_an_error_status_is_a_failure_that_names_it_and_a_new_port(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health = HealthResult(HealthKind.ANSWERED, 404)
        machine.holder = PortHolder(77, "workerd")

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.level is Level.FAIL
        assert finding.title.endswith("(HTTP 404)")
        assert finding.details == (f"Port {PORT} is held by workerd (pid 77).",)
        assert finding.fix == (
            f"Quit that program, or move the dashboard: set `port` in the config file to {PORT + 1} (free now), "
            "then run `usagedash agent install` and `usagedash open`."
        )

    def test_the_remedy_names_the_environment_variable_when_this_shell_sets_it(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", str(PORT))
        machine.health = HealthResult(HealthKind.ANSWERED, 404)

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.fix == (
            "Quit that program, or move the dashboard: this shell sets USAGEDASH_PORT, which outranks the config "
            f"file, so change it to {PORT + 1} (free now), then run `usagedash agent install` and `usagedash open`."
        )

    def test_the_remedy_at_the_highest_port_is_a_port_the_settings_accept(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health = HealthResult(HealthKind.ANSWERED, 404)

        finding = only(doctor.diagnose_agent(paths, doctor.MAX_PORT), "Something other than the dashboard")

        assert finding.fix is not None
        assert "to 65534 (free now)" in finding.fix
        assert "65536" not in finding.fix

    def test_the_remedy_asks_for_any_free_port_when_none_nearby_is_free(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor, "_port_is_free", lambda _port: False)
        machine.health = HealthResult(HealthKind.TIMED_OUT)

        finding = only(doctor.diagnose_agent(paths, PORT), "did not answer")

        assert finding.fix is not None
        assert "to a free port from 1024 to 65535, then run" in finding.fix

    def test_a_healthy_dashboard_never_probes_for_a_free_port(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(_port: int) -> bool:
            raise AssertionError("a free port is looked for only when the port is in the way")

        monkeypatch.setattr(doctor, "_port_is_free", never)

        doctor.diagnose_agent(paths, PORT)

    def test_a_running_agent_that_is_not_listening_yet_is_told_to_wait_and_look_at_the_logs(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health = HealthResult(HealthKind.REFUSED)

        finding = only(doctor.diagnose_agent(paths, PORT), "nothing is listening")

        assert finding.level is Level.FAIL
        assert finding.details == ("It may still be starting.",)
        assert finding.fix == (
            "Wait ten seconds, then run `usagedash doctor` again. If it stays like this, the log lines below "
            "say why it could not start."
        )

    @pytest.mark.parametrize(
        ("health", "title"),
        [
            (
                HealthResult(HealthKind.ANSWERED, 500),
                "The agent (pid 838) holds port 9123, but its health check answers HTTP 500",
            ),
            (
                HealthResult(HealthKind.TIMED_OUT),
                "The agent (pid 838) holds port 9123, but it did not answer within 2 seconds",
            ),
            (
                HealthResult(HealthKind.FAILED),
                "The agent (pid 838) holds port 9123, but the health request failed before an answer",
            ),
        ],
    )
    def test_an_agent_that_holds_the_port_but_answers_badly_is_restarted_not_moved(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        health: HealthResult,
        title: str,
    ) -> None:
        def never(_port: int) -> bool:
            raise AssertionError("a different port cannot help an agent that holds this one")

        monkeypatch.setattr(doctor, "_port_is_free", never)
        machine.health, machine.holder = health, PortHolder(838, "python3.13")

        finding = only(doctor.diagnose_agent(paths, PORT), "holds port")

        assert finding.level is Level.FAIL
        assert finding.title == title
        assert finding.details == ("No other program is in the way, so a different port will not help.",)
        assert finding.fix == "Restart it with `usagedash agent install`. If this comes back, read the log lines below."

    def test_a_listener_that_is_not_the_agent_is_still_blamed_and_moved_off_when_it_answers_badly(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health, machine.holder = HealthResult(HealthKind.ANSWERED, 500), PortHolder(99, "python3.13")

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.details == (f"Port {PORT} is held by python3.13 (pid 99).",)
        assert finding.fix is not None
        assert "move the dashboard" in finding.fix

    def test_a_holder_that_could_not_be_seen_is_not_assumed_to_be_the_agent(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health, machine.holder = HealthResult(HealthKind.TIMED_OUT), None

        finding = only(doctor.diagnose_agent(paths, PORT), "accepted a connection")

        assert finding.fix is not None
        assert "move the dashboard" in finding.fix

    def test_nothing_listening_with_no_agent_process_points_back_at_the_agent_problem(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state, machine.health = CRASHED, HealthResult(HealthKind.REFUSED)

        finding = only(doctor.diagnose_agent(paths, PORT), "Nothing is listening")

        assert finding.fix == "Fix the agent problem above first."

    def test_a_listener_that_never_answers_is_a_failure(self, paths: agent.AgentPaths, machine: Machine) -> None:
        machine.health = HealthResult(HealthKind.TIMED_OUT)
        machine.holder = PortHolder(5, "other")

        finding = only(doctor.diagnose_agent(paths, PORT), "did not answer")

        assert finding.level is Level.FAIL
        assert finding.details == (f"Port {PORT} is held by other (pid 5).",)

    def test_a_request_that_fails_outright_is_a_failure(self, paths: agent.AgentPaths, machine: Machine) -> None:
        machine.health = HealthResult(HealthKind.FAILED)

        assert only(doctor.diagnose_agent(paths, PORT), "failed before an answer").level is Level.FAIL

    def test_no_port_means_no_request_is_made(self, paths: agent.AgentPaths, machine: Machine) -> None:
        doctor.diagnose_agent(paths, None)

        assert machine.probed == []


class TestTheFullDiagnosis:
    @pytest.fixture(autouse=True)
    def a_quiet_machine(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(agent, "agent_path_omissions", lambda _executables: [])
        monkeypatch.setattr(doctor, "checked_executable", lambda executable, _purpose: str(executable))
        # Why: a bare CLI name is looked up on the agent's PATH, which falls back to this shell's when no
        # agent is installed. This machine's own PATH may well hold the real CLIs.
        monkeypatch.setenv("PATH", str(tmp_path / "no-clis-here"))

    def test_a_healthy_installation_has_no_warning_no_failure_and_no_log_section(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        settings = settings_for(tmp_path)
        write_log(paths.server_log, "2026-11-01 INFO usagedash.api: collected\n")

        findings = doctor.diagnose(paths, shell=settings, served=settings)

        assert not [finding for finding in findings if finding.level in (Level.WARN, Level.FAIL)]
        assert not [title for title in titles(findings) if title.startswith("Last lines")]
        assert doctor.has_failure(findings) is False

    def test_a_warning_alone_does_not_add_the_log_lines_to_the_report(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment = None
        write_log(paths.server_log, "2026-11-01 INFO usagedash.api: collected\n")

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert titles(findings, Level.WARN) == ["launchd lists an agent whose property list is gone"]
        assert titles(findings, Level.FAIL) == []
        assert not [title for title in titles(findings) if title.startswith("Last lines")]

    def test_two_logs_that_end_the_same_way_are_shown_once(self, paths: agent.AgentPaths) -> None:
        mirrored = "2026-11-01 ERROR usagedash.cli: could not bind\n"
        write_log(paths.launchd_logs[1], mirrored)
        write_log(paths.server_log, mirrored)

        findings = doctor.log_findings(paths)

        assert titles(findings) == [f"Last lines of {paths.launchd_logs[1]}"]

    def test_two_logs_that_differ_are_both_shown(self, paths: agent.AgentPaths) -> None:
        write_log(paths.launchd_logs[1], "Traceback (most recent call last):\n")
        write_log(paths.server_log, "2026-11-01 ERROR usagedash.cli: could not bind\n")

        findings = doctor.log_findings(paths)

        assert titles(findings) == [
            f"Last lines of {paths.launchd_logs[1]}",
            f"Last lines of {paths.server_log}",
        ]

    def test_a_problem_adds_the_log_lines_to_the_report(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = CRASHED
        write_log(paths.server_log, "2026-11-01 ERROR usagedash.cli: could not bind\n")

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "usagedash.log").details == ("2026-11-01 ERROR usagedash.cli: could not bind",)
        assert doctor.has_failure(findings) is True

    def test_settings_that_do_not_load_are_a_failure_and_the_agent_is_still_checked(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=ConfigError("port must be between 1024 and 65535, got 80"), served=None)

        assert titles(findings, Level.FAIL) == [
            "The settings cannot be loaded: port must be between 1024 and 65535, got 80",
            "The agent is not installed",
        ]

    def test_settings_the_installed_agent_cannot_read_are_a_failure(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        findings = doctor.diagnose(
            paths, shell=settings_for(tmp_path), served=ConfigError("port must be an integer, got 'x'")
        )

        assert only(findings, "installed agent's settings").level is Level.FAIL

    def test_the_agents_port_wins_over_this_shells_and_the_difference_is_reported(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        findings = doctor.diagnose(paths, shell=settings_for(tmp_path, 9123), served=settings_for(tmp_path, 9200))

        assert machine.probed == ["http://127.0.0.1:9200/healthz"]
        assert only(findings, "serves port 9200").level is Level.INFO

    def test_the_same_port_everywhere_is_not_reported_as_a_difference(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        settings = settings_for(tmp_path)

        assert "serves port" not in " ".join(titles(doctor.diagnose(paths, shell=settings, served=settings)))

    def test_with_no_settings_at_all_nothing_is_probed(self, paths: agent.AgentPaths, machine: Machine) -> None:
        doctor.diagnose(paths, shell=None, served=None)

        assert machine.probed == []

    def test_a_data_directory_another_account_could_replace_is_a_failure(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(_directory: Path) -> None:
            raise ConfigError("/shared/x is writable by other accounts")

        monkeypatch.setattr(doctor, "ensure_no_ancestor_lets_another_account_replace", refuse)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "data directory cannot be trusted").level is Level.FAIL

    def test_a_data_directory_that_does_not_exist_yet_is_information_and_is_not_created(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "does not exist yet").level is Level.INFO
        assert not (tmp_path / "data").exists()

    def test_an_existing_data_directory_is_ok(self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine) -> None:
        (tmp_path / "data").mkdir()

        assert (
            only(doctor.diagnose(paths, shell=settings_for(tmp_path), served=None), "is safe to use").level is Level.OK
        )

    def test_a_symbolic_link_at_the_data_path_is_a_failure_even_when_it_points_at_a_folder(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (tmp_path / "data").symlink_to(elsewhere)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        failure = only(findings, "is a symbolic link")
        assert failure.level is Level.FAIL
        assert failure.fix is not None
        assert "USAGEDASH_DATA_DIR" in failure.fix
        assert "is safe to use" not in " ".join(titles(findings))

    def test_a_file_at_the_data_path_is_a_failure(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").write_text("")

        findings = doctor.data_directory_findings(settings)

        assert titles(findings, Level.FAIL) == [f"The data directory {tmp_path / 'data'} is a file, not a folder"]

    def test_settings_that_cannot_be_read_because_a_file_is_in_the_way_are_reported_as_a_settings_failure(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        (tmp_path / "data").write_text("")
        outcome = _settings_outcome_for(tmp_path)

        findings = doctor.diagnose(paths, shell=outcome, served=None)

        assert isinstance(outcome, ConfigError)
        assert only(findings, "The settings cannot be loaded").level is Level.FAIL

    def test_a_data_directory_that_belongs_to_another_account_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir()
        owner = os.stat(data).st_uid
        monkeypatch.setattr(doctor, "ensure_no_ancestor_lets_another_account_replace", lambda _directory: None)
        monkeypatch.setattr(doctor.os, "geteuid", lambda: owner + 1)

        findings = doctor.data_directory_findings(settings)

        assert titles(findings, Level.FAIL) == [f"The data directory {data} belongs to another account"]
        assert "is safe to use" not in " ".join(titles(findings))

    @pytest.mark.parametrize("mode", [0o777, 0o770, 0o707, 0o722, 0o702])
    def test_a_data_directory_this_account_owns_but_others_can_write_is_a_failure_not_safe_to_use(
        self, tmp_path: Path, mode: int
    ) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir()
        data.chmod(mode)

        try:
            findings = doctor.data_directory_findings(settings)
        finally:
            data.chmod(0o700)

        failure = only(findings, "Another account can change the data directory")
        assert failure.level is Level.FAIL
        assert failure.title.startswith(f"Another account can change the data directory {data}: ")
        assert failure.details == (
            "It holds the database and the access token, so another account could replace them.",
        )
        assert failure.fix == "Run `usagedash agent install`, which makes it owner-only again."
        assert "is safe to use" not in " ".join(titles(findings))

    @pytest.mark.parametrize("mode", [0o700, 0o750, 0o755, 0o705, 0o500])
    def test_a_data_directory_others_can_at_most_read_is_safe_to_use(self, tmp_path: Path, mode: int) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir()
        data.chmod(mode)

        try:
            findings = doctor.data_directory_findings(settings)
        finally:
            data.chmod(0o700)

        assert titles(findings, Level.OK) == [f"The data directory {data} is safe to use"]

    def test_an_access_control_entry_that_lets_another_account_change_the_directory_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(
            doctor, "what_lets_another_account_change", lambda _path: "an access-control entry lets another write"
        )

        findings = doctor.data_directory_findings(settings)

        assert titles(findings, Level.FAIL) == [
            f"Another account can change the data directory {tmp_path / 'data'}: "
            "an access-control entry lets another write"
        ]

    def test_a_data_directory_whose_permissions_cannot_be_read_is_a_failure_that_says_what_to_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").mkdir()

        def unreadable(path: Path) -> str | None:
            raise ConfigError(f"cannot examine {path}: PermissionError")

        monkeypatch.setattr(doctor, "what_lets_another_account_change", unreadable)

        findings = doctor.data_directory_findings(settings)

        failure = only(findings, "cannot be examined")
        assert (
            failure.title
            == f"The data directory cannot be examined: cannot examine {tmp_path / 'data'}: PermissionError"
        )
        assert (
            failure.fix
            == "Check the permissions of that folder and the ones above it, then run `usagedash doctor` again."
        )

    def test_a_data_directory_this_account_owns_is_safe_to_use(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(doctor, "ensure_no_ancestor_lets_another_account_replace", lambda _directory: None)

        assert titles(doctor.data_directory_findings(settings), Level.OK) == [
            f"The data directory {tmp_path / 'data'} is safe to use"
        ]

    def test_a_data_directory_that_cannot_be_examined_is_a_failure_named_by_its_error_type(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = tmp_path / "data"
        real_lstat = os.lstat

        def lstat(path: str | Path) -> os.stat_result:
            if Path(path) == data:
                raise PermissionError
            return real_lstat(path)

        monkeypatch.setattr(doctor, "ensure_no_ancestor_lets_another_account_replace", lambda _directory: None)
        monkeypatch.setattr(doctor.os, "lstat", lstat)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert (
            only(findings, "cannot be examined").title
            == f"The data directory {data} cannot be examined: PermissionError"
        )

    def test_the_binaries_the_agent_was_installed_with_are_checked_not_the_ones_this_shell_finds(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        pinned = executable_file(tmp_path / "pinned-claude")
        served = replace(settings_for(tmp_path), binaries=Binaries(str(pinned), "codex", "grok"))

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=served)

        assert only(findings, "claude: ").title == f"claude: {pinned.resolve()}"
        assert "claude: not found on PATH" not in titles(findings)

    def test_the_installed_binaries_are_still_checked_when_this_shells_settings_do_not_load(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        findings = doctor.diagnose(paths, shell=ConfigError("port must be an integer"), served=settings_for(tmp_path))

        assert only(findings, "claude: not found").level is Level.INFO
        assert only(findings, "The settings cannot be loaded").level is Level.FAIL

    def test_a_bare_cli_name_is_found_on_the_path_the_agent_runs_with(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        on_agent_path = executable_file(tmp_path / "agent-bin" / "claude")
        machine.environment = {"PATH": str(on_agent_path.parent)}

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert only(findings, "claude: ").title == f"claude: {on_agent_path.resolve()}"
        assert only(findings, "codex: not found").level is Level.INFO

    def test_without_an_installed_agent_a_bare_name_is_looked_up_on_this_shells_path(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_shell_path = executable_file(tmp_path / "shell-bin" / "codex")
        monkeypatch.setenv("PATH", str(on_shell_path.parent))
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "codex: ").title == f"codex: {on_shell_path.resolve()}"

    def test_a_pin_whose_file_was_removed_is_judged_by_the_replacement_the_agent_finds(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A Homebrew cask upgrade deletes the version directory the pin names; the agent then looks the
        # vendor command up on its own PATH (README, "A recorded file that has been removed").
        current = executable_file(tmp_path / "bin" / "codex")
        removed = tmp_path / "Caskroom" / "0.165.0" / "bin" / "codex-0.165.0"
        machine.environment = {"PATH": str(current.parent)}
        served = replace(settings_for(tmp_path), binaries=Binaries("claude", str(removed), "grok"))

        with caplog.at_level("WARNING"):
            finding = only(doctor.diagnose(paths, shell=None, served=served), "codex: ")

        assert finding.level is Level.WARN
        assert finding.title == f"codex: the recorded {removed} is gone; the agent uses {current.resolve()}"
        assert finding.fix == "Run `usagedash agent install` to record the current file."
        assert caplog.messages == []
        assert logging.root.manager.disable == 0

    def test_a_pin_whose_file_was_removed_with_no_replacement_is_a_warning_that_says_to_reinstall(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        removed = tmp_path / "gone" / "claude"
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(removed), "codex", "grok"))

        finding = only(doctor.diagnose(paths, shell=settings, served=None), "claude: ")

        assert finding.level is Level.WARN
        assert finding.title.startswith(f"claude: cannot use {removed}: FileNotFoundError")
        assert finding.fix == "Install claude again, then run `usagedash agent install`."

    def test_a_cli_link_that_points_at_nothing_is_a_warning_not_an_ok(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        dangling = tmp_path / "claude"
        dangling.symlink_to(tmp_path / "removed")
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(dangling), "codex", "grok"))

        finding = only(doctor.diagnose(paths, shell=settings, served=None), "claude: ")

        assert finding.level is Level.WARN
        assert finding.title == f"claude: {tmp_path.resolve() / 'removed'} is not a file"
        assert finding.fix == "Install claude again, then run `usagedash agent install`."

    def test_a_cli_that_is_a_folder_is_a_warning_not_an_ok(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        folder = tmp_path / "claude"
        folder.mkdir()
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(folder), "codex", "grok"))

        assert only(doctor.diagnose(paths, shell=settings, served=None), "claude: ").level is Level.WARN

    def test_a_cli_without_the_execute_bit_is_a_warning_that_names_the_chmod(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        plain = executable_file(tmp_path / "claude", 0o644)
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(plain), "codex", "grok"))

        finding = only(doctor.diagnose(paths, shell=settings, served=None), "claude: ")

        assert finding.level is Level.WARN
        assert finding.title == f"claude: {plain.resolve()} is not executable"
        assert finding.fix == f"Run `chmod u+x {plain.resolve()}`, then `usagedash agent install`."

    def test_a_vendor_cli_that_is_not_installed_is_information_not_a_problem(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        for name in ("claude", "codex", "grok"):
            assert only(findings, f"{name}: not found").level is Level.INFO

    def test_a_cli_another_account_could_change_is_a_warning_and_a_good_one_is_ok(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        good = executable_file(tmp_path / "good")
        bad = executable_file(tmp_path / "bad", 0o777)
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(good), str(bad), "grok"))

        findings = doctor.diagnose(paths, shell=settings, served=None)

        assert only(findings, "claude: ").level is Level.OK
        unsafe = only(findings, "codex: ")
        assert unsafe.level is Level.WARN
        assert "its permissions (0777) let other accounts write" in unsafe.title

    def test_a_vendor_cli_directory_left_off_the_path_is_a_warning_with_its_reason(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        omission = agent.PathOmission("/opt/tools/bin", "its permissions (0775) let other accounts write", True)
        monkeypatch.setattr(agent, "agent_path_omissions", lambda _executables: [omission])

        finding = only(doctor.diagnose(paths, shell=settings_for(tmp_path), served=None), "/opt/tools/bin")

        assert finding.level is Level.WARN
        assert finding.details == ("its permissions (0775) let other accounts write",)

    @pytest.mark.parametrize(
        ("count", "noun", "hidden"), [(1, "directory", 0), (2, "directories", 0), (11, "directories", 1)]
    )
    def test_other_omitted_directories_are_summarized_and_capped(
        self,
        tmp_path: Path,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        count: int,
        noun: str,
        hidden: int,
    ) -> None:
        omissions = [agent.PathOmission(f"/missing/{index}", "missing", False) for index in range(count)]
        monkeypatch.setattr(agent, "agent_path_omissions", lambda _executables: omissions)

        finding = only(doctor.diagnose(paths, shell=settings_for(tmp_path), served=None), "would be left off")

        assert finding.title == f"{count} PATH {noun} would be left off the agent's PATH"
        assert finding.level is Level.INFO
        listed = [detail for detail in finding.details if detail.startswith("/missing/")]
        assert len(listed) == min(count, doctor.MAX_LISTED_OMISSIONS)
        assert (f"... and {hidden} more" in finding.details) is (hidden > 0)

    def test_a_path_that_cannot_be_examined_is_a_warning(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(_executables: list[str]) -> list[agent.PathOmission]:
            raise PermissionError

        monkeypatch.setattr(agent, "agent_path_omissions", unreadable)

        finding = only(doctor.diagnose(paths, shell=settings_for(tmp_path), served=None), "could not be examined")

        assert finding.title == "This shell's PATH could not be examined: PermissionError"

    def test_another_operating_system_is_a_failure_and_stops_the_system_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor.sys, "platform", "linux")

        findings = doctor.system_findings()

        assert titles(findings) == ["This is linux, not macOS"]
        assert findings[0].level is Level.FAIL

    def test_a_python_that_install_would_refuse_is_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(_executable: Path, _purpose: str) -> str:
            raise ConfigError("cannot use Python executable /x/python: its permissions (0777)")

        monkeypatch.setattr(doctor.sys, "platform", "darwin")
        monkeypatch.setattr(doctor, "checked_executable", refuse)

        findings = doctor.system_findings()

        assert titles(findings, Level.FAIL) == [
            "`agent install` would refuse this Python: cannot use Python executable /x/python: its permissions (0777)"
        ]

    def test_a_usable_python_is_reported_as_information(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.sys, "platform", "darwin")

        findings = doctor.system_findings()

        assert [finding.level for finding in findings] == [Level.INFO]
        assert findings[0].title.startswith("Python 3.")


class TestTheReport:
    def test_findings_render_with_their_details_and_fix_and_end_in_a_verdict(self, tmp_path: Path) -> None:
        findings = [
            Finding(Level.OK, "fine"),
            Finding(Level.FAIL, "broken", ("because",), "do this"),
        ]

        text = doctor.render(findings, tmp_path)

        assert text.splitlines()[:4] == ["[ok  ] fine", "[FAIL] broken", "       because", "       -> do this"]
        assert "Result: 1 problem found." in text
        assert doctor.ISSUE_HINT in text

    def test_the_verdict_counts_problems_warnings_and_checks_and_ignores_information(self, tmp_path: Path) -> None:
        info = Finding(Level.INFO, "i")
        assert doctor.render([info], tmp_path).endswith(
            "Result: all 0 checks passed. Run `usagedash open` to view the dashboard."
        )
        assert doctor.render([Finding(Level.OK, "a"), info], tmp_path).endswith(
            "Result: all 1 checks passed. Run `usagedash open` to view the dashboard."
        )
        assert doctor.render([Finding(Level.WARN, "w"), Finding(Level.OK, "a")], tmp_path).endswith(
            "Result: working, with 1 warning. 2 checks ran."
        )
        assert doctor.render([Finding(Level.WARN, "w")] * 2, tmp_path).endswith(
            "Result: working, with 2 warnings. 2 checks ran."
        )
        assert "Result: 2 problems found." in doctor.render([Finding(Level.FAIL, "f", fix="do this")] * 2, tmp_path)

    def test_a_warning_alone_is_not_a_failure(self) -> None:
        assert doctor.has_failure([Finding(Level.WARN, "w"), Finding(Level.OK, "o")]) is False

    @pytest.mark.parametrize("fix", [None, ""])
    def test_a_failure_that_names_no_next_step_cannot_be_built(self, fix: str | None) -> None:
        with pytest.raises(ValueError, match="a FAIL must name the next step: broken"):
            Finding(Level.FAIL, "broken", ("because",), fix)

    @pytest.mark.parametrize("level", [Level.OK, Level.INFO, Level.WARN])
    def test_a_finding_that_is_not_a_failure_needs_no_next_step(self, level: Level) -> None:
        assert Finding(level, "fine").fix is None

    def test_the_home_folder_is_shown_as_a_tilde_only_where_it_is_a_whole_path_prefix(self) -> None:
        home = Path("/Users/greg")
        text = "/Users/greg/Library /Users/greg2/x /Users/greg /Users/greg.old/y at /Users/greg."

        assert doctor.abbreviate_home(text, home) == "~/Library /Users/greg2/x ~ /Users/greg.old/y at ~."

    def test_an_installed_failure_report_shows_the_agent_checks_alone(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = CRASHED
        write_log(paths.launchd_logs[1], f"{tmp_path}/venv/bin/python: no module named usagedash\n")

        text = doctor.render_agent_failure(paths, PORT, tmp_path)

        assert "[FAIL] The agent started and stopped (exit code 1)" in text
        assert "~/venv/bin/python: no module named usagedash" in text
        assert str(tmp_path) not in text
        assert "Result:" not in text


class TestReadingALog:
    def test_the_last_lines_of_a_log_are_returned(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("".join(f"line {number}\n" for number in range(30)))

        assert doctor.tail_lines(log, max_lines=3) == ["line 27", "line 28", "line 29"]

    def test_a_missing_log_is_none_and_an_empty_one_is_an_empty_list(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.write_text("")

        assert doctor.tail_lines(tmp_path / "absent") is None
        assert doctor.tail_lines(empty) == []

    def test_a_line_cut_in_the_middle_by_the_byte_limit_is_dropped(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("0123456789\nwhole one\nwhole two\n")

        assert doctor.tail_lines(log, max_bytes=19) == ["whole two"]

    def test_a_cut_that_lands_exactly_on_a_line_start_keeps_that_line(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("0123456789\nwhole one\nwhole two\n")

        assert doctor.tail_lines(log, max_bytes=20) == ["whole one", "whole two"]

    def test_a_log_that_fits_the_byte_limit_keeps_its_first_line(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("first\nsecond\n")

        assert doctor.tail_lines(log, max_bytes=len("first\nsecond\n")) == ["first", "second"]

    def test_a_link_is_not_followed(self, tmp_path: Path) -> None:
        target, link = tmp_path / "target", tmp_path / "link"
        target.write_text("secret\n")
        link.symlink_to(target)

        assert doctor.tail_lines(link) is None

    def test_a_pipe_and_a_directory_are_not_read(self, tmp_path: Path) -> None:
        pipe = tmp_path / "pipe"
        os.mkfifo(pipe)

        assert doctor.tail_lines(pipe) is None
        assert doctor.tail_lines(tmp_path) is None

    def test_a_file_that_belongs_to_another_account_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "log"
        log.write_text("line\n")
        monkeypatch.setattr(doctor.os, "geteuid", lambda: os.stat(log).st_uid + 1)

        assert doctor.tail_lines(log) is None

    def test_a_log_that_cannot_be_read_is_none_and_its_descriptor_is_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "log"
        log.write_text("line\n")
        closed: list[int] = []
        real_close = os.close

        def failing_read(_descriptor: int, _length: int, _offset: int) -> bytes:
            raise OSError("input/output error")

        def recording_close(descriptor: int) -> None:
            closed.append(descriptor)
            real_close(descriptor)

        monkeypatch.setattr(doctor.os, "pread", failing_read)
        monkeypatch.setattr(doctor.os, "close", recording_close)

        assert doctor.tail_lines(log) is None
        assert len(closed) == 1

    def test_control_characters_never_reach_the_terminal(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_bytes(b"ok\x1b[31mred\x00\tx\n")

        assert doctor.tail_lines(log) == ["ok?[31mred? x"]

    def test_a_very_long_line_is_cut(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("y" * 1000 + "\n")

        assert doctor.tail_lines(log) == ["y" * doctor.MAX_LOG_LINE_CHARACTERS + "…"]


class TestFindingWhoHoldsAPort:
    def test_lsof_is_asked_for_the_listener_and_its_first_process_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout="p838\ncpython3.13\np900\ncother\n")
        monkeypatch.setattr(doctor.subprocess, "run", run)

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")
        assert run.commands == [[doctor.LSOF_BINARY, "-nP", "-iTCP:8787", "-sTCP:LISTEN", "-Fpc"]]

    @pytest.mark.parametrize("output", ["", "pabc\ncfoo\n", "p12\n", "cfoo\n", "garbage"])
    def test_output_that_does_not_name_a_process_means_none_is_found(
        self, monkeypatch: pytest.MonkeyPatch, output: str
    ) -> None:
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(returncode=1, stdout=output))

        assert doctor.port_holder(8787) is None

    def test_a_missing_or_stuck_lsof_means_none_is_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for failure in (FileNotFoundError("lsof"), doctor.subprocess.TimeoutExpired(["lsof"], 10)):
            monkeypatch.setattr(doctor.subprocess, "run", FakeRun(error=failure))
            assert doctor.port_holder(8787) is None

    def test_a_command_name_is_made_printable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout="p5\nca\x1bb\n"))

        assert doctor.port_holder(8787) == PortHolder(5, "a?b")


class Status(BaseHTTPRequestHandler):
    code = 200

    def do_GET(self) -> None:
        self.send_response(self.code)
        self.send_header("Location", "http://127.0.0.1:1/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def serve_status() -> Iterator[Callable[[int], str]]:
    servers: list[ThreadingHTTPServer] = []

    def start(code: int) -> str:
        handler = type("Answers", (Status,), {"code": code})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/healthz"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


class TestProbingHealth:
    @pytest.mark.parametrize("code", [200, 302, 404, 500])
    def test_the_status_the_listener_answers_is_reported_and_no_redirect_is_followed(
        self, serve_status: Callable[[int], str], code: int
    ) -> None:
        assert doctor.probe_health(serve_status(code)) == HealthResult(HealthKind.ANSWERED, code)

    def test_a_port_nothing_listens_on_is_refused(self) -> None:
        with socket.socket() as closed:
            closed.bind(("127.0.0.1", 0))
            port = closed.getsockname()[1]

        assert doctor.probe_health(f"http://127.0.0.1:{port}/healthz") == HealthResult(HealthKind.REFUSED)

    def test_a_listener_that_never_answers_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "HEALTH_TIMEOUT_SECONDS", 0.3)
        with socket.socket() as stalled:
            stalled.bind(("127.0.0.1", 0))
            stalled.listen()
            port = stalled.getsockname()[1]

            assert doctor.probe_health(f"http://127.0.0.1:{port}/healthz") == HealthResult(HealthKind.TIMED_OUT)

    def test_a_listener_that_does_not_speak_http_is_a_failed_request(self) -> None:
        with socket.socket() as garbage:
            garbage.bind(("127.0.0.1", 0))
            garbage.listen()
            port = garbage.getsockname()[1]

            def reply() -> None:
                connection, _ = garbage.accept()
                with connection:
                    connection.recv(1024)
                    connection.sendall(b"not http at all\r\n\r\n")

            thread = threading.Thread(target=reply, daemon=True)
            thread.start()

            assert doctor.probe_health(f"http://127.0.0.1:{port}/healthz") == HealthResult(HealthKind.FAILED)
            thread.join(timeout=5)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USAGEDASH_DATA_DIR", str(tmp_path / "data"))
    for name in ("claude", "codex", "grok"):
        monkeypatch.setenv(f"USAGEDASH_{name.upper()}_BINARY", name)
    monkeypatch.delenv("USAGEDASH_CONFIG", raising=False)
    monkeypatch.delenv("USAGEDASH_PORT", raising=False)
    monkeypatch.setattr(agent, "account_home", lambda: tmp_path)
    monkeypatch.setattr(agent, "agent_path_omissions", lambda _executables: [])
    monkeypatch.setattr(doctor, "checked_executable", lambda executable, _purpose: str(executable))
    return tmp_path


class TestTheDoctorCommand:
    def test_a_healthy_installation_exits_zero_and_creates_nothing(
        self, home: Path, machine: Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {}

        assert cli.main(["doctor"]) == cli.EXIT_OK

        output = capsys.readouterr().out
        assert "Result: all" in output
        assert "FAIL" not in output
        assert not (home / "data").exists()

    def test_a_config_file_other_accounts_can_read_is_made_owner_only_as_every_command_does(
        self, home: Path, machine: Machine
    ) -> None:
        machine.environment = {}
        data = home / "data"
        data.mkdir(mode=0o700)
        config = data / "config.toml"
        config.write_text("poll_interval_seconds = 300\n")
        config.chmod(0o644)

        cli.main(["doctor"])

        assert stat.S_IMODE(config.stat().st_mode) == 0o600
        assert config.read_text() == "poll_interval_seconds = 300\n"

    def test_a_config_file_other_accounts_can_write_is_refused_and_left_as_it_is(
        self, home: Path, machine: Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {}
        data = home / "data"
        data.mkdir(mode=0o700)
        config = data / "config.toml"
        config.write_text("poll_interval_seconds = 300\n")
        config.chmod(0o666)

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "[FAIL] The settings cannot be loaded: cannot use" in output
        assert "If it says a file's permissions are unsafe, make that file writable by you alone" in output
        assert "(`chmod go-w <file>`)" in output
        assert stat.S_IMODE(config.stat().st_mode) == 0o666

    def test_a_problem_exits_one_and_never_prints_the_home_folder(
        self, home: Path, machine: Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment, machine.state = None, None
        machine.health = HealthResult(HealthKind.REFUSED)

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "[FAIL] The agent is not installed" in output
        assert "Result: 2 problems found." in output
        assert str(home) not in output

    def test_invalid_settings_are_reported_and_do_not_stop_the_other_checks(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("USAGEDASH_PORT", "80")
        machine.environment, machine.state = None, None

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "[FAIL] The settings cannot be loaded: port must be between 1024 and 65535" in output
        assert "[FAIL] The agent is not installed" in output

    def test_the_port_the_agent_was_installed_with_is_the_one_asked(
        self, home: Path, machine: Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {"USAGEDASH_PORT": "9456"}

        cli.main(["doctor"])

        assert machine.probed == ["http://127.0.0.1:9456/healthz"]
        assert "The installed agent serves port 9456; this shell would use 8787" in capsys.readouterr().out

    def test_installed_settings_that_do_not_load_are_reported_not_raised(
        self, home: Path, machine: Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {"USAGEDASH_PORT": "80"}

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        assert "[FAIL] The installed agent's settings cannot be read: port must be between" in capsys.readouterr().out


class TestWaitingForTheNewAgent:
    URL = "http://127.0.0.1:8787/"

    @dataclass
    class Clock:
        now: float = 0.0
        sleeps: list[float] = field(default_factory=list)

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

    @pytest.fixture
    def clock(self) -> "TestWaitingForTheNewAgent.Clock":
        return self.Clock()

    @staticmethod
    def answers(monkeypatch: pytest.MonkeyPatch, *results: bool) -> list[str]:
        asked: list[str] = []
        remaining = list(results)

        def listening(url: str) -> bool:
            asked.append(url)
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]

        monkeypatch.setattr(cli, "_dashboard_is_listening", listening)
        return asked

    def wait(self, paths: agent.AgentPaths, clock: "TestWaitingForTheNewAgent.Clock") -> bool:
        return cli._wait_until_answering(paths, self.URL, clock.monotonic, clock.sleep)

    def test_an_agent_that_answers_at_once_is_not_waited_for(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asked = self.answers(monkeypatch, True)

        assert self.wait(paths, clock) is True
        assert asked == [self.URL]
        assert clock.sleeps == []

    def test_a_slow_start_is_waited_out_in_half_second_steps(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        machine.state = agent.ServiceState("running", 838, 1, "(never exited)", None)
        self.answers(monkeypatch, False, False, False, True)

        assert self.wait(paths, clock) is True
        assert clock.sleeps == [cli.INSTALL_READY_POLL_SECONDS] * 3

    def test_an_agent_that_never_answers_is_given_up_on_at_the_deadline(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.answers(monkeypatch, False)

        assert self.wait(paths, clock) is False
        assert clock.now == cli.INSTALL_READY_WAIT_SECONDS

    def test_an_agent_whose_run_already_failed_ends_the_wait_at_once(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        machine.state = CRASHED
        self.answers(monkeypatch, False)

        assert self.wait(paths, clock) is False
        assert clock.sleeps == []

    def test_a_first_run_that_has_not_exited_is_not_taken_for_a_failure(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        machine.state = agent.ServiceState("spawn scheduled", None, 1, "(never exited)", None)
        self.answers(monkeypatch, False, True)

        assert self.wait(paths, clock) is True

    def test_launchd_that_cannot_be_asked_does_not_end_the_wait(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise agent.AgentError("launchctl print failed (1): boom")

        monkeypatch.setattr(agent, "service_state", unreachable)
        self.answers(monkeypatch, False, True)

        assert self.wait(paths, clock) is True

    def test_an_agent_launchd_no_longer_lists_does_not_end_the_wait(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        machine.state = None
        self.answers(monkeypatch, False, True)

        assert self.wait(paths, clock) is True


class TestInstallReportsWhetherTheDashboardAnswers:
    @pytest.fixture
    def installs(self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
        recorded: list[tuple[object, ...]] = []
        monkeypatch.setattr(agent, "install", lambda *arguments: recorded.append(arguments))
        return recorded

    def test_an_answering_agent_is_reported_installed_and_exits_zero(
        self,
        installs: list[tuple[object, ...]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(cli, "_wait_until_answering", lambda _paths, _url: True)

        assert cli.main(["agent", "install"]) == cli.EXIT_OK

        out = capsys.readouterr().out
        assert "waiting for it to answer at http://127.0.0.1:8787/" in out
        assert out.rstrip().endswith("on http://127.0.0.1:8787; run `usagedash open` to view it")
        assert len(installs) == 1

    def test_an_agent_that_does_not_answer_exits_one_says_so_and_shows_what_was_found(
        self,
        installs: list[tuple[object, ...]],
        machine: Machine,
        home: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        machine.state = CRASHED
        machine.health = HealthResult(HealthKind.REFUSED)
        write_log(agent.agent_paths().launchd_logs[1], f"{home}/.venv/bin/python: can't open file usagedash\n")
        monkeypatch.setattr(cli, "_wait_until_answering", lambda _paths, _url: False)

        assert cli.main(["agent", "install"]) == cli.EXIT_FAILURE

        captured = capsys.readouterr()
        assert "usagedash open" not in captured.out
        assert "is installed, but the dashboard is not answering at http://127.0.0.1:8787/" in captured.err
        assert "[FAIL] The agent started and stopped (exit code 1)" in captured.err
        assert "~/.venv/bin/python: can't open file usagedash" in captured.err
        assert str(home) not in captured.err
        assert captured.err.rstrip().endswith("Run `usagedash doctor` for the full check.")
        assert len(installs) == 1


class TestAgentStatus:
    @pytest.fixture
    def label(self, home: Path) -> str:
        return agent.agent_paths().label

    def test_an_agent_launchd_does_not_list_is_one_line_and_launchd_is_not_asked_for_more(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def not_asked(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise AssertionError("a service that is not loaded has no state to ask for")

        monkeypatch.setattr(agent, "is_loaded", lambda _paths: False)
        monkeypatch.setattr(agent, "service_state", not_asked)

        assert cli.main(["agent", "status"]) == cli.EXIT_OK

        assert capsys.readouterr().out == f"{label}: not loaded\n"

    def test_a_running_agent_shows_its_pid_runs_logs_and_the_doctor_command(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(agent, "service_state", lambda _paths: RUNNING)

        assert cli.main(["agent", "status"]) == cli.EXIT_OK

        assert capsys.readouterr().out == (
            f"{label}: running (pid 838)\n"
            "  runs: 1\n"
            f"  logs: {agent.agent_paths().log_dir}\n"
            "  Run `usagedash doctor` to check that it answers.\n"
        )

    def test_a_loaded_agent_that_crashed_is_not_called_running(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(agent, "service_state", lambda _paths: CRASHED)

        assert cli.main(["agent", "status"]) == cli.EXIT_OK

        first = capsys.readouterr().out.splitlines()[0]
        assert first == f"{label}: loaded, not running; the last run ended with exit code 1"
        assert "running (pid" not in first

    def test_a_loaded_agent_killed_by_a_signal_is_reported_by_the_signal_not_an_exit_code(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(
            agent, "service_state", lambda _paths: agent.ServiceState("waiting", None, 2, "(never exited)", "Killed: 9")
        )

        cli.main(["agent", "status"])

        first = capsys.readouterr().out.splitlines()[0]
        assert first == f"{label}: loaded, not running; the last run ended with signal Killed: 9"
        assert "exit code" not in first

    def test_a_loaded_agent_with_no_process_and_no_failure_names_launchds_state(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(
            agent, "service_state", lambda _paths: agent.ServiceState("waiting", None, None, None, None)
        )

        cli.main(["agent", "status"])

        assert capsys.readouterr().out.splitlines()[0] == f"{label}: loaded, not running (waiting)"

    def test_an_agent_that_vanished_between_the_two_questions_is_not_called_running(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "is_loaded", lambda _paths: True)
        monkeypatch.setattr(agent, "service_state", lambda _paths: None)

        cli.main(["agent", "status"])

        assert capsys.readouterr().out.splitlines()[0] == f"{label}: loaded, not running (state not stated)"
