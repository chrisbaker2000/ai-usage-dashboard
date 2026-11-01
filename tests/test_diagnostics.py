"""Self-serve diagnostics: launchd state parsing, `usagedash doctor`, and the install-time readiness wait.

Nothing here talks to this machine's launchd or to a listener it did not start: launchd and the port
holder are faked, and the health probe runs against sockets opened inside the test.
"""

import logging
import os
import re
import socket
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.fakes.fake_run import FakeRun
from usagedash import agent, cli, doctor, executables
from usagedash.config import Binaries, Settings, load_settings
from usagedash.doctor import Finding, HealthKind, HealthResult, Level, PortHolder, SettingsOutcome
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
        [
            ("838", 838),
            ("1", 1),
            ("999999999999999999", 999999999999999999),
            ("0", None),
            ("-5", None),
            ("-1", None),
            ("", None),
            ("-", None),
            ("12abc", None),
            ("١٢", None),
            ("1.5", None),
            ("+5", None),
            ("1" * 19, None),
            ("1" * 80 + "x", None),
            ("1" * 5000, None),
        ],
    )
    def test_a_process_id_is_read_only_when_it_is_plainly_a_positive_number(
        self, text: str, expected: int | None
    ) -> None:
        assert agent.parse_service_state(f"{TAB}pid = {text}\n").pid == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("0", 0), ("4", 4), ("-1", None), ("x", None), ("", None), ("1" * 19, None), ("1" * 80 + "x", None)],
    )
    def test_a_run_count_is_read_only_when_it_is_plainly_a_number_of_zero_or_more(
        self, text: str, expected: int | None
    ) -> None:
        assert agent.parse_service_state(f"{TAB}runs = {text}\n").runs == expected

    @pytest.mark.parametrize("text", ["0", "-5", "1" * 19, "1" * 80 + "x", "1" * 5000, "abc", "", "-"])
    def test_a_pid_line_that_is_not_a_process_id_leaves_the_process_unknown_whatever_the_state_says(
        self, text: str
    ) -> None:
        for state in ("running", "spawn scheduled", "waiting"):
            parsed = agent.parse_service_state(f"{TAB}state = {state}\n{TAB}pid = {text}\n")

            assert parsed.pid is None
            assert parsed.pid_unreadable is True
            assert parsed.status is agent.AgentStatus.UNKNOWN

    def test_no_pid_line_with_a_state_other_than_running_is_still_a_stopped_agent(self) -> None:
        parsed = agent.parse_service_state(f"{TAB}state = waiting\n")

        assert parsed.pid_unreadable is False
        assert parsed.status is agent.AgentStatus.STOPPED

    def test_a_readable_pid_is_a_running_agent_and_not_unreadable(self) -> None:
        parsed = agent.parse_service_state(f"{TAB}state = running\n{TAB}pid = 838\n")

        assert parsed.pid == 838
        assert parsed.pid_unreadable is False
        assert parsed.status is agent.AgentStatus.RUNNING

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

    def test_a_text_field_longer_than_the_limit_is_not_understood_and_is_never_cut_into_one_that_is(self) -> None:
        limit = agent.MAX_SERVICE_FIELD_LENGTH

        assert agent.parse_service_state(f"{TAB}state = {'x' * limit}\n").state == "x" * limit
        assert agent.parse_service_state(f"{TAB}state = {'x' * (limit + 1)}\n").state is None
        assert agent.parse_service_state(f"{TAB}state = {'x' * 500}\n").state is None

    def test_a_last_exit_and_a_last_signal_longer_than_the_limit_are_not_understood_either(self) -> None:
        limit = agent.MAX_SERVICE_FIELD_LENGTH
        listing = f"{TAB}last exit code = {'1' * (limit + 1)}\n{TAB}last terminating signal = {'x' * (limit + 1)}\n"

        parsed = agent.parse_service_state(listing)

        assert parsed.last_exit is None
        assert parsed.last_signal is None
        assert parsed.last_run_failed is False

    def test_a_value_is_judged_whole_so_eighty_digits_and_a_letter_are_not_a_number(self) -> None:
        limit = agent.MAX_SERVICE_FIELD_LENGTH
        parsed = agent.parse_service_state(f"{TAB}pid = {'1' * limit}x\n{TAB}runs = {'2' * limit}x\n")

        assert (parsed.pid, parsed.runs) == (None, None)

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

    def test_a_vendor_cli_directory_whose_path_holds_the_separator_is_an_omission_and_not_a_silent_skip(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "a:b"
        self.executable(directory, "claude")

        omissions = agent.agent_path_omissions([str(directory / "claude")], "")

        assert [omission for omission in omissions if omission.directory == str(directory)] == [
            agent.PathOmission(str(directory), "its path holds the PATH separator", holds_vendor_cli=True)
        ]

    def test_a_relative_entry_resolved_into_a_folder_whose_name_holds_the_separator_is_an_omission(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "x:y"
        (work / "bin").mkdir(parents=True)
        monkeypatch.chdir(work)

        omissions = agent.agent_path_omissions([], "bin")

        assert [
            (omission.directory, omission.holds_vendor_cli) for omission in omissions if "x:y" in omission.directory
        ] == [(str(Path.cwd() / "bin"), False)]

    def test_the_install_warning_names_a_directory_whose_path_holds_the_separator_and_leaves_it_off(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path / "a:b"
        self.executable(directory, "claude")

        with caplog.at_level("WARNING", logger="usagedash.agent"):
            search_path = agent.agent_search_path([str(directory / "claude")], "")

        assert str(directory) not in search_path.split(os.pathsep)
        assert f"{directory}: its path holds the PATH separator" in " ".join(caplog.messages)

    def test_a_directory_is_reported_once_however_often_it_is_named(self, tmp_path: Path) -> None:
        directory = tmp_path / "a:b"
        self.executable(directory, "claude")
        pins = [str(directory / "claude"), str(directory / "claude")]

        omissions = agent.agent_path_omissions(pins, "")

        assert [omission.directory for omission in omissions].count(str(directory)) == 1

    def test_doctor_names_a_vendor_cli_directory_the_agent_path_cannot_hold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "a:b"
        self.executable(directory, "claude")
        monkeypatch.setattr(agent, "helper_problems", lambda _executables, _directory, _shell_path=None: [])
        settings = replace(settings_for(tmp_path), binaries=Binaries(str(directory / "claude"), "codex", "grok"))

        [finding] = [
            found
            for found in doctor.path_findings(settings)
            if found.title.startswith("The directory of a vendor CLI") and str(directory) in found.title
        ]

        assert finding.level is Level.WARN
        assert finding.details[0] == "its path holds the PATH separator"


class TestRequiredHelpers:
    """What would stop ``agent install`` from supplying `security`, found without changing anything."""

    @pytest.fixture
    def helper(self, tmp_path: Path) -> Path:
        return executable_file(tmp_path / "usr-bin" / "security")

    @staticmethod
    def problems(helper: Path, working: Path, search_path: str = "") -> list[str]:
        return executables.required_helper_problems(search_path, working, [helper])

    def test_a_helper_already_on_the_path_needs_nothing(self, tmp_path: Path, helper: Path) -> None:
        assert self.problems(helper, tmp_path / "data", str(helper.parent)) == []

    def test_a_safe_helper_the_path_lacks_is_supplied_with_no_problem_when_nothing_is_prepared_yet(
        self, tmp_path: Path, helper: Path
    ) -> None:
        assert self.problems(helper, tmp_path / "data") == []

    def test_a_helper_that_is_not_there_is_a_problem_that_names_it(self, tmp_path: Path) -> None:
        absent = tmp_path / "usr-bin" / "security"
        absent.parent.mkdir()

        assert self.problems(absent, tmp_path / "data") == [f"cannot use required helper {absent}: FileNotFoundError"]

    def test_a_helper_without_the_execute_bit_is_a_problem(self, tmp_path: Path) -> None:
        helper = executable_file(tmp_path / "usr-bin" / "security", mode=0o644)

        [problem] = self.problems(helper, tmp_path / "data")

        assert "not an executable regular file" in problem

    @pytest.mark.parametrize("error", [OSError("input/output error"), RuntimeError("symlink loop")])
    def test_an_error_while_resolving_the_helper_is_a_problem_named_by_its_type(
        self, tmp_path: Path, helper: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def fail(_executable: Path, _purpose: str) -> str:
            raise error

        monkeypatch.setattr(executables, "checked_executable", fail)

        assert self.problems(helper, tmp_path / "data") == [f"cannot use a required helper: {type(error).__name__}"]

    def test_a_private_folder_already_holding_the_right_link_is_no_problem(self, tmp_path: Path, helper: Path) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(helper.resolve())

        assert self.problems(helper, working) == []

    def test_a_private_folder_with_no_link_in_it_yet_is_no_problem(self, tmp_path: Path, helper: Path) -> None:
        working = tmp_path / "data"
        (working / "agent-bin").mkdir(parents=True, mode=0o700)

        assert self.problems(helper, working) == []

    def test_a_link_that_names_another_target_is_a_problem(self, tmp_path: Path, helper: Path) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(tmp_path / "somewhere-else")

        assert self.problems(helper, working) == [
            f"cannot use required helper {folder / 'security'}: it names an unexpected target"
        ]

    def test_a_private_folder_another_account_could_change_is_a_problem(self, tmp_path: Path, helper: Path) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True)
        folder.chmod(0o777)
        try:
            problems = self.problems(helper, working)
        finally:
            folder.chmod(0o700)

        assert len(problems) == 1
        assert str(folder) in problems[0]

    def test_a_private_folder_that_is_not_a_folder_is_a_problem_named_by_its_type(
        self, tmp_path: Path, helper: Path
    ) -> None:
        working = tmp_path / "data"
        working.mkdir()
        folder = working / "agent-bin"
        folder.write_text("")
        folder.chmod(0o600)

        assert self.problems(helper, working) == [f"cannot use {folder}: NotADirectoryError"]

    def test_a_regular_file_where_the_link_belongs_is_a_problem_named_by_its_type(
        self, tmp_path: Path, helper: Path
    ) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").write_text("")

        assert self.problems(helper, working) == [f"cannot prepare required helpers in {folder}: OSError"]

    def test_a_folder_that_cannot_be_looked_at_is_a_problem_named_by_its_type(
        self, tmp_path: Path, helper: Path
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("")

        assert self.problems(helper, blocker / "data") == [
            f"cannot prepare required helpers in {blocker / 'data' / 'agent-bin'}: NotADirectoryError"
        ]

    def test_an_unsafe_entry_beside_the_link_is_a_problem_because_install_checks_every_entry_in_the_folder(
        self, tmp_path: Path, helper: Path
    ) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(helper.resolve())
        sibling = executable_file(folder / "loose", mode=0o777)

        [problem] = self.problems(helper, working)

        assert "executable entry" in problem
        assert str(sibling) in problem

    @pytest.mark.parametrize("folder_exists", [False, True])
    def test_a_data_directory_whose_path_holds_the_path_separator_is_a_problem_before_anything_exists(
        self, tmp_path: Path, helper: Path, folder_exists: bool
    ) -> None:
        working = tmp_path / "a:b" / "data"
        if folder_exists:
            (working / "agent-bin").mkdir(parents=True, mode=0o700)

        assert self.problems(helper, working) == ["the required helper directory holds the PATH separator"]

    def test_nothing_is_created_when_asking(self, tmp_path: Path, helper: Path) -> None:
        working = tmp_path / "data"

        self.problems(helper, working)

        assert not working.exists()

    @pytest.mark.parametrize(
        "scenario",
        ["clean", "unexpected target", "unsafe sibling", "separator", "no helper", "read-only folder"],
    )
    def test_install_refuses_exactly_what_the_check_reports(self, tmp_path: Path, helper: Path, scenario: str) -> None:
        if scenario == "read-only folder" and os.geteuid() == 0:
            pytest.skip("root can write to a folder with no write bit")
        working = tmp_path / ("a:b" if scenario == "separator" else "data")
        working.mkdir()
        folder = working / "agent-bin"
        if scenario in ("unexpected target", "unsafe sibling", "read-only folder"):
            folder.mkdir(mode=0o700)
        if scenario == "unexpected target":
            (folder / "security").symlink_to(tmp_path / "somewhere-else")
        if scenario == "unsafe sibling":
            executable_file(folder / "loose", mode=0o777)
        if scenario == "no helper":
            helper.chmod(0o644)
        if scenario == "read-only folder":
            folder.chmod(0o500)

        try:
            reported = self.problems(helper, working)

            if reported:
                with pytest.raises(ConfigError) as refused:
                    executables.with_required_helpers("", working, [helper])
                assert [str(refused.value)] == reported
            else:
                assert executables.with_required_helpers("", working, [helper]).endswith(str(folder.resolve()))
        finally:
            if folder.exists():
                folder.chmod(0o700)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a folder with no write bit")
    def test_a_helper_folder_this_account_cannot_write_to_is_a_problem_when_the_link_is_missing(
        self, tmp_path: Path, helper: Path
    ) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        folder.chmod(0o500)
        try:
            problems = self.problems(helper, working)
        finally:
            folder.chmod(0o700)

        assert problems == [f"cannot prepare required helpers in {folder}: PermissionError"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a folder with no write bit")
    def test_a_read_only_helper_folder_that_already_holds_the_right_link_is_no_problem(
        self, tmp_path: Path, helper: Path
    ) -> None:
        working = tmp_path / "data"
        folder = working / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(helper.resolve())
        folder.chmod(0o500)
        try:
            problems = self.problems(helper, working)
        finally:
            folder.chmod(0o700)

        assert problems == []

    def test_the_agent_asks_with_the_path_install_builds_and_changes_nothing(
        self, tmp_path: Path, helper: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (helper,))
        working = tmp_path / "data"
        helper.parent.chmod(0o700)

        on_the_path = agent.helper_problems([], working, str(helper.parent))
        off_the_path = agent.helper_problems([], working, str(tmp_path))

        assert on_the_path == []
        assert off_the_path == []
        assert not working.exists()

    def test_the_agent_reports_the_problem_when_the_path_lacks_an_unsafe_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        absent = tmp_path / "nowhere" / "security"
        monkeypatch.setattr(agent, "REQUIRED_PATH_HELPERS", (absent,))

        [problem] = agent.helper_problems([], tmp_path / "data", str(tmp_path))

        assert problem == f"cannot use {absent.parent}: FileNotFoundError"

    def test_doctor_names_what_install_would_refuse_and_the_command_to_run_after_fixing_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            agent, "helper_problems", lambda _executables, _directory, _shell_path=None: ["cannot use boom"]
        )

        [finding] = [found for found in doctor.path_findings(settings_for(tmp_path)) if "helper" in found.title]

        assert finding.level is Level.WARN
        assert finding.title == "`agent install` would refuse a helper Claude needs: cannot use boom"
        assert finding.fix == "Fix what that names, then run `usagedash agent install`."

    def test_doctor_says_nothing_about_helpers_when_install_would_supply_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent, "helper_problems", lambda _executables, _directory, _shell_path=None: [])

        assert not [found for found in doctor.path_findings(settings_for(tmp_path)) if "helper" in found.title]

    def test_a_path_that_cannot_be_examined_is_one_warning_for_both_questions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(_executables: object, _directory: object, _shell_path: object = None) -> list[str]:
            raise PermissionError

        monkeypatch.setattr(agent, "helper_problems", fail)

        findings = doctor.path_findings(settings_for(tmp_path))

        assert titles(findings) == ["This shell's PATH could not be examined: PermissionError"]


@dataclass
class Machine:
    """What launchd, the port, and the network say, for the checks that ask them."""

    environment: dict[str, str] | None = field(default_factory=lambda: {"USAGEDASH_PORT": str(PORT)})
    state: agent.ServiceState | None = RUNNING
    health: HealthResult = field(default_factory=lambda: HealthResult(HealthKind.ANSWERED, 200))
    holder: PortHolder | None = None
    probed: list[str] = field(default_factory=list)

    def nothing_installed(self) -> None:
        """No property list, nothing loaded in launchd, nobody on the port."""
        self.environment, self.state, self.holder = None, None, None

    def running(self) -> None:
        """An agent installed with a property list, running as pid 838, and the process holding the port."""
        self.environment = {"USAGEDASH_PORT": str(PORT)}
        self.state = RUNNING
        self.holder = PortHolder(838, "python3.13")


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
    monkeypatch.setattr(agent, "helper_problems", lambda _executables, _directory, _shell_path=None: [])
    monkeypatch.setattr(executables, "REQUIRED_PATH_HELPERS", ())
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


def agents_config_file(home: Path) -> Path:
    """Where the installed agent's config file is when its plist records none: the default under the home."""
    return home / "Library" / "Application Support" / "ai-usage-dashboard" / "config.toml"


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

    def test_a_service_whose_property_list_is_gone_is_a_failure_beside_the_running_finding(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment = None

        findings = doctor.diagnose_agent(paths, None)

        gone = only(findings, "property list is gone")
        assert gone.level is Level.FAIL
        assert gone.details == ("Which port, data folder, and CLIs it uses cannot be known without it.",)
        assert gone.fix == "Run `usagedash agent install` from the repository directory."
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

    def test_a_listed_agent_that_launchd_says_is_not_running_and_has_not_failed_is_a_warning(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = agent.ServiceState("waiting", None, 1, "(never exited)", None)

        finding = only(doctor.diagnose_agent(paths, None), "is not running it")

        assert finding.level is Level.WARN
        assert finding.title == "launchd lists the agent but is not running it"

    @pytest.mark.parametrize(
        "state",
        [
            agent.ServiceState(None, None, None, None, None),
            agent.ServiceState("running", None, 1, "(never exited)", None),
            agent.ServiceState("running", None, 3, "1", None),
        ],
        ids=["nothing stated", "says running without a pid", "says running, last run failed"],
    )
    def test_a_listing_that_does_not_say_whether_the_agent_runs_is_not_called_not_running(
        self, paths: agent.AgentPaths, machine: Machine, state: agent.ServiceState
    ) -> None:
        machine.state = state

        findings = doctor.diagnose_agent(paths, None)

        finding = only(findings, "does not say whether it is running")
        assert finding.level is Level.WARN
        assert finding.title == "launchd lists the agent, but its listing does not say whether it is running"
        assert "is not running it" not in " ".join(titles(findings))
        assert "started and stopped" not in " ".join(titles(findings))

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


AGENT_CASES: list[tuple[str, agent.ServiceState | None, bool, doctor.AgentStatus]] = [
    ("not loaded", None, True, doctor.AgentStatus.STOPPED),
    ("waiting", agent.ServiceState("waiting", None, 1, "(never exited)", None), True, doctor.AgentStatus.STOPPED),
    ("crashed", CRASHED, True, doctor.AgentStatus.STOPPED),
    ("running", RUNNING, True, doctor.AgentStatus.RUNNING),
    ("state not stated", agent.ServiceState(None, None, None, None, None), True, doctor.AgentStatus.UNKNOWN),
    ("running without a pid", agent.ServiceState("running", None, 1, None, None), True, doctor.AgentStatus.UNKNOWN),
    ("launchd unreachable", None, False, doctor.AgentStatus.UNKNOWN),
]
HOLDERS: dict[str, PortHolder | None] = {
    "nobody seen": None,
    "the agent's pid": PortHolder(838, "python3.13"),
    "another pid": PortHolder(99, "workerd"),
}
HEALTHS: dict[str, HealthResult] = {
    "200": HealthResult(HealthKind.ANSWERED, 200),
    "404": HealthResult(HealthKind.ANSWERED, 404),
    "500": HealthResult(HealthKind.ANSWERED, 500),
    "refused": HealthResult(HealthKind.REFUSED),
    "timed out": HealthResult(HealthKind.TIMED_OUT),
    "failed": HealthResult(HealthKind.FAILED),
}


class TestNothingIsClaimedThatTheEvidenceDoesNotAllow:
    """Every combination of what launchd says, who holds the port, and what the health request got.

    Each input can be unknown, and an unknown input must never count as evidence for either side. The
    findings that blame a party or clear one are checked against what is known, not against a table.
    """

    @pytest.fixture(autouse=True)
    def every_nearby_port_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "_port_is_free", lambda _port: True)
        monkeypatch.delenv("USAGEDASH_PORT", raising=False)

    @pytest.mark.parametrize("agent_case", AGENT_CASES, ids=[case[0] for case in AGENT_CASES])
    @pytest.mark.parametrize("holder_id", HOLDERS)
    @pytest.mark.parametrize("health_id", HEALTHS)
    def test_a_finding_blames_or_clears_a_party_only_as_far_as_the_evidence_goes(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        agent_case: tuple[str, agent.ServiceState | None, bool, doctor.AgentStatus],
        holder_id: str,
        health_id: str,
    ) -> None:
        _, state, reachable, status = agent_case
        if reachable:
            machine.state = state
        else:

            def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
                raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

            monkeypatch.setattr(agent, "service_state", unreachable)
        holder, health = HOLDERS[holder_id], HEALTHS[health_id]
        machine.holder, machine.health = holder, health

        answers = [finding for finding in doctor.diagnose_agent(paths, PORT) if str(PORT) in finding.title]

        assert len(answers) == 1
        answer = answers[0]
        stopped = status is doctor.AgentStatus.STOPPED
        running = status is doctor.AgentStatus.RUNNING
        unknown = status is doctor.AgentStatus.UNKNOWN
        agent_holds = running and holder is not None and holder.pid == 838
        other_holds = holder is not None and (stopped or (running and holder.pid != 838))
        healthy = health.kind is HealthKind.ANSWERED and health.status == 200
        title = answer.title

        assert ("the agent is not running" in title) <= stopped
        assert (
            "Something other than the dashboard" in title
            or "is not the one holding it" in title
            or "something else is serving it" in title
        ) <= (stopped or other_holds)
        assert title.startswith("The agent (pid") <= agent_holds
        holder_unseen = holder is None
        assert ("could not tell whether the agent is serving it" in title) == (
            healthy and (unknown or (running and holder_unseen))
        )
        assert (answer.level is Level.OK) <= (healthy and agent_holds)
        if healthy and stopped:
            assert answer.level is Level.FAIL
        if healthy and running:
            expected = Level.FAIL if other_holds else Level.OK if agent_holds else Level.WARN
            assert answer.level is expected
        if not healthy and health.kind is not HealthKind.REFUSED and agent_holds:
            assert "Restart it with `usagedash agent install`" in (answer.fix or "")
            assert "move the dashboard" not in (answer.fix or "")
        if not healthy and health.kind is not HealthKind.REFUSED and not agent_holds and (stopped or other_holds):
            assert "move the dashboard" in (answer.fix or "")
        if (
            not healthy
            and health.kind is not HealthKind.REFUSED
            and not agent_holds
            and not stopped
            and not other_holds
        ):
            assert (answer.fix or "").startswith("Restart the agent with `usagedash agent install`")


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
        [None, agent.ServiceState("waiting", None, 1, "(never exited)", None), CRASHED],
        ids=["not loaded", "loaded and waiting", "crashed"],
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
        machine.state, machine.holder = agent.ServiceState("waiting", None, 1, "(never exited)", None), None

        finding = only(doctor.diagnose_agent(paths, PORT), "the agent is not running")

        assert finding.level is Level.FAIL
        assert not [detail for detail in finding.details if detail.startswith("Port ")]

    @pytest.mark.parametrize(
        "state",
        [agent.ServiceState(None, None, None, None, None), agent.ServiceState("running", None, 1, None, None)],
        ids=["state not stated", "says running without a pid"],
    )
    def test_a_healthy_answer_is_not_blamed_on_another_process_when_launchds_state_does_not_say(
        self, paths: agent.AgentPaths, machine: Machine, state: agent.ServiceState
    ) -> None:
        machine.state, machine.holder = state, PortHolder(838, "python3.13")

        findings = doctor.diagnose_agent(paths, PORT)

        finding = only(findings, "could not tell whether the agent is serving it")
        assert finding.level is Level.WARN
        assert finding.details == (
            "launchd's state for the agent could not be read.",
            f"Port {PORT} is held by python3.13 (pid 838).",
        )
        assert (
            finding.fix == "Run `usagedash agent status`. If the agent is not running, run `usagedash agent install`."
        )
        assert "the agent is not running, so something else" not in " ".join(titles(findings))
        assert "Something other than the dashboard" not in " ".join(titles(findings))

    def test_a_healthy_answer_is_not_blamed_on_another_process_when_launchd_cannot_be_asked(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

        monkeypatch.setattr(agent, "service_state", unreachable)
        machine.holder = PortHolder(99, "python3.13")

        findings = doctor.diagnose_agent(paths, PORT)

        assert only(findings, "launchd cannot be asked").level is Level.FAIL
        assert only(findings, "could not tell whether the agent is serving it").level is Level.WARN
        assert "the agent is not running, so something else" not in " ".join(titles(findings))

    def test_the_report_does_not_end_in_working_when_the_agent_is_not_the_one_serving(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state = agent.ServiceState("waiting", None, 1, "(never exited)", None)
        machine.holder = PortHolder(99, "python3.13")

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
        machine.health, machine.holder = HealthResult(HealthKind.ANSWERED, 404), PortHolder(77, "workerd")

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.fix == (
            "Quit that program, or move the dashboard: this shell sets USAGEDASH_PORT, which outranks the config "
            f"file, so change it to {PORT + 1} (free now), then run `usagedash agent install` and `usagedash open`."
        )

    @pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
    def test_the_remedy_names_the_config_file_when_the_variable_is_unset_or_empty_because_both_mean_unset(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is None:
            monkeypatch.delenv("USAGEDASH_PORT", raising=False)
        else:
            monkeypatch.setenv("USAGEDASH_PORT", value)
        machine.health, machine.holder = HealthResult(HealthKind.ANSWERED, 404), PortHolder(77, "workerd")

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.fix is not None
        assert "set `port` in the config file to" in finding.fix
        assert "USAGEDASH_PORT" not in finding.fix

    def test_the_remedy_at_the_highest_port_is_a_port_the_settings_accept(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health, machine.holder = HealthResult(HealthKind.ANSWERED, 404), PortHolder(77, "workerd")

        finding = only(doctor.diagnose_agent(paths, doctor.MAX_PORT), "Something other than the dashboard")

        assert finding.fix is not None
        assert "to 65534 (free now)" in finding.fix
        assert "65536" not in finding.fix

    def test_the_remedy_asks_for_any_free_port_when_none_nearby_is_free(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor, "_port_is_free", lambda _port: False)
        machine.health, machine.holder = HealthResult(HealthKind.TIMED_OUT), PortHolder(77, "workerd")

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

    @pytest.mark.parametrize(
        ("health", "title"),
        [
            (HealthResult(HealthKind.ANSWERED, 500), f"Port {PORT} answers HTTP 500, not the dashboard's 200"),
            (
                HealthResult(HealthKind.TIMED_OUT),
                f"Port {PORT} accepted a connection but did not answer within 2 seconds",
            ),
            (HealthResult(HealthKind.FAILED), f"The request to port {PORT} failed before an answer"),
        ],
    )
    def test_a_running_agent_and_a_holder_nobody_could_see_blames_neither_and_restarts_first(
        self, paths: agent.AgentPaths, machine: Machine, health: HealthResult, title: str
    ) -> None:
        machine.health, machine.holder = health, None

        failures = [finding for finding in doctor.diagnose_agent(paths, PORT) if finding.level is Level.FAIL]

        assert len(failures) == 1
        finding = failures[0]
        assert finding.title == title
        assert finding.details == (
            "Doctor could not tell which process holds the port, so it cannot tell the agent from another program.",
        )
        assert finding.fix is not None
        assert finding.fix.startswith(
            "Restart the agent with `usagedash agent install`, then run `usagedash doctor` again. If it fails the "
            f"same way, another program may hold port {PORT}. Quit that program, or move the dashboard: "
        )
        assert finding.fix.endswith("The log lines below may say more.")
        assert "Something other than the dashboard" not in finding.title

    def test_no_agent_process_and_no_visible_holder_that_answers_badly_is_another_program(
        self, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.state, machine.health, machine.holder = None, HealthResult(HealthKind.ANSWERED, 404), None

        finding = only(doctor.diagnose_agent(paths, PORT), "Something other than the dashboard")

        assert finding.level is Level.FAIL
        assert finding.details == ()
        assert finding.fix is not None
        assert finding.fix.startswith("Quit that program, or move the dashboard: ")

    @pytest.mark.parametrize(
        "state",
        [None, CRASHED, agent.ServiceState(None, None, None, None, None)],
        ids=["not loaded", "crashed", "state not stated"],
    )
    def test_nothing_listening_without_a_running_agent_has_a_remedy_of_its_own_not_a_pointer_to_another_finding(
        self, paths: agent.AgentPaths, machine: Machine, state: agent.ServiceState | None
    ) -> None:
        machine.state, machine.health = state, HealthResult(HealthKind.REFUSED)

        finding = only(doctor.diagnose_agent(paths, PORT), "Nothing is listening")

        assert finding.level is Level.FAIL
        assert finding.fix == (
            "Run `usagedash agent install`, then `usagedash doctor` again. If nothing listens after that, the "
            "log lines below say why."
        )
        assert "above" not in finding.fix

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
    def a_quiet_machine(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: Machine) -> None:
        # Why nothing installed by default: `served=None` means there is no property list, and this shell's
        # settings stand in only then. A test about a running agent says so with `machine.running()`.
        machine.nothing_installed()
        monkeypatch.setattr(agent, "agent_path_omissions", lambda _executables: [])
        monkeypatch.setattr(doctor, "checked_executable", lambda executable, _purpose: str(executable))
        # Why: a bare CLI name is looked up on the agent's PATH, which falls back to this shell's when no
        # agent is installed. This machine's own PATH may well hold the real CLIs.
        monkeypatch.setenv("PATH", str(tmp_path / "no-clis-here"))

    def test_a_healthy_installation_has_no_warning_no_failure_and_no_log_section(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.running()
        settings = settings_for(tmp_path)
        write_log(paths.server_log, "2026-11-01 INFO usagedash.api: collected\n")

        findings = doctor.diagnose(paths, shell=settings, served=settings)

        assert not [finding for finding in findings if finding.level in (Level.WARN, Level.FAIL)]
        assert not [title for title in titles(findings) if title.startswith("Last lines")]
        assert doctor.has_failure(findings) is False

    def test_a_warning_alone_does_not_add_the_log_lines_to_the_report(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.running()
        write_log(paths.server_log, "2026-11-01 INFO usagedash.api: collected\n")
        gone = tmp_path / "gone" / "claude"
        served = replace(settings_for(tmp_path), binaries=Binaries(str(gone), "codex", "grok"))

        findings = doctor.diagnose(paths, shell=served, served=served)

        warnings = titles(findings, Level.WARN)
        assert len(warnings) == 1
        assert warnings[0].startswith(f"claude: cannot use {gone}")
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

    def test_launchds_standard_output_capture_is_shown_too_so_a_failure_reported_only_there_is_not_missed(
        self, paths: agent.AgentPaths
    ) -> None:
        standard_output, standard_error = paths.launchd_logs
        write_log(standard_output, "fatal: only on standard output\n")
        write_log(standard_error, "Traceback (most recent call last):\n")

        findings = doctor.log_findings(paths)

        assert titles(findings) == [f"Last lines of {standard_error}", f"Last lines of {standard_output}"]
        assert findings[1].details == ("fatal: only on standard output",)

    def test_a_tail_repeated_by_a_later_log_is_not_shown_again_even_with_another_log_between_them(
        self, paths: agent.AgentPaths
    ) -> None:
        standard_output, standard_error = paths.launchd_logs
        mirrored = "2026-11-01 ERROR usagedash.cli: could not bind\n"
        write_log(standard_error, mirrored)
        write_log(standard_output, "something else entirely\n")
        write_log(paths.server_log, mirrored)

        findings = doctor.log_findings(paths)

        assert titles(findings) == [f"Last lines of {standard_error}", f"Last lines of {standard_output}"]

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

    def test_installed_settings_that_cannot_be_read_are_not_replaced_by_this_shells_in_the_checks_that_follow(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.running()

        findings = doctor.diagnose(
            paths, shell=settings_for(tmp_path, 9123), served=ConfigError("port must be an integer, got 'x'")
        )

        assert machine.probed == []
        assert only(findings, "installed agent's settings").level is Level.FAIL
        assert not [title for title in titles(findings) if "data directory" in title]
        assert not [title for title in titles(findings) if title.startswith(("claude:", "codex:", "grok:"))]
        assert not [
            title
            for title in titles(findings)
            if title.startswith(("The dashboard answers", "Nothing is listening", "Port 9123", "The request to port"))
        ]
        assert "launchd is running the agent (pid 838)" in titles(findings)

    @pytest.mark.parametrize("agent_case", AGENT_CASES, ids=[case[0] for case in AGENT_CASES])
    @pytest.mark.parametrize("plist", ["present", "present but invalid", "absent", "untrusted"])
    @pytest.mark.parametrize("shell_loads", [True, False], ids=["shell settings load", "shell settings do not load"])
    @pytest.mark.parametrize("health_id", ["200", "refused"])
    @pytest.mark.parametrize("holder_id", ["nobody seen", "the agent's pid"])
    def test_a_report_without_a_failure_has_established_that_the_dashboard_answers(
        self,
        tmp_path: Path,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        agent_case: tuple[str, agent.ServiceState | None, bool, agent.AgentStatus],
        plist: str,
        shell_loads: bool,
        health_id: str,
        holder_id: str,
    ) -> None:
        case_id, state, reachable, _ = agent_case
        if reachable:
            machine.state = state
        else:

            def unreachable(_paths: agent.AgentPaths) -> agent.ServiceState | None:
                raise agent.AgentError("launchctl print could not be run: FileNotFoundError")

            monkeypatch.setattr(agent, "service_state", unreachable)
        served: SettingsOutcome = None
        if plist == "untrusted":

            def untrusted(_paths: agent.AgentPaths) -> dict[str, str] | None:
                raise agent.AgentError("another account could change it")

            monkeypatch.setattr(agent, "installed_environment", untrusted)
            served = agent.AgentError("another account could change it")
        elif plist == "absent":
            machine.environment = None
        elif plist == "present but invalid":
            machine.environment = {"USAGEDASH_PORT": "80"}
            served = ConfigError("port must be between 1024 and 65535, got 80")
        else:
            machine.environment = {"USAGEDASH_PORT": str(PORT)}
            served = settings_for(tmp_path)
        machine.health, machine.holder = HEALTHS[health_id], HOLDERS[holder_id]
        shell = settings_for(tmp_path) if shell_loads else ConfigError("port must be an integer")

        findings = doctor.diagnose(paths, shell=shell, served=served)

        if not doctor.has_failure(findings):
            assert machine.probed == [f"http://127.0.0.1:{PORT}/healthz"]
            assert health_id == "200"
        nothing_installed = plist == "absent" and case_id == "not loaded"
        if machine.probed and not isinstance(served, Settings):
            assert shell_loads
            assert nothing_installed
        for finding in findings:
            if finding.level is Level.FAIL:
                assert finding.fix
                assert "above" not in finding.fix

    def test_a_loaded_agent_with_no_property_list_is_not_diagnosed_with_this_shells_settings(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.running()
        machine.environment = None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path, 9123), served=None)

        assert machine.probed == []
        assert only(findings, "property list is gone").level is Level.FAIL
        assert doctor.has_failure(findings) is True
        assert not [title for title in titles(findings) if "data directory" in title]
        assert not [title for title in titles(findings) if title.startswith(("claude:", "codex:", "grok:"))]

    def test_this_shells_settings_stand_in_only_when_no_agent_is_installed(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path, 9123), served=None)

        assert machine.probed == ["http://127.0.0.1:9123/healthz"]
        assert any("data directory" in title for title in titles(findings))

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

    def test_a_missing_data_directory_whose_nearest_folder_cannot_be_written_to_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = tmp_path / "data"
        asked: list[tuple[Path, int]] = []

        def cannot_write(path: Path, mode: int) -> bool:
            asked.append((Path(path), mode))
            return False

        monkeypatch.setattr(doctor.os, "access", cannot_write)

        [failure] = doctor.data_directory_findings(settings_for(tmp_path))

        assert failure.level is Level.FAIL
        assert failure.title == (
            f"The data directory {data} does not exist and cannot be made: {tmp_path} is the nearest folder "
            "that exists, and this account cannot write to it"
        )
        assert failure.fix == (
            "Make that a folder you can write to, or point USAGEDASH_DATA_DIR at one, "
            "then run `usagedash agent install`."
        )
        assert asked == [(tmp_path, os.W_OK | os.X_OK)]
        assert not data.exists()

    def test_the_nearest_folder_that_exists_is_the_one_asked_when_several_levels_are_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = load_settings(
            {"USAGEDASH_DATA_DIR": str(tmp_path / "a" / "b" / "data"), "USAGEDASH_PORT": str(PORT)}, home=tmp_path
        )
        (tmp_path / "a").mkdir(mode=0o700)
        asked: list[Path] = []

        def cannot_write(path: Path, _mode: int) -> bool:
            asked.append(Path(path))
            return False

        monkeypatch.setattr(doctor.os, "access", cannot_write)

        [failure] = doctor.data_directory_findings(settings)

        assert asked == [tmp_path / "a"]
        assert f"{tmp_path / 'a'} is the nearest folder that exists" in failure.title

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a folder with no write bit")
    def test_a_read_only_parent_stops_the_data_directory_from_being_made_for_real(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o700)
        settings = load_settings(
            {"USAGEDASH_DATA_DIR": str(locked / "data"), "USAGEDASH_PORT": str(PORT)}, home=tmp_path
        )
        locked.chmod(0o500)
        try:
            [finding] = doctor.data_directory_findings(settings)
        finally:
            locked.chmod(0o700)

        assert finding.level is Level.FAIL
        assert "this account cannot write to it" in finding.title

    def test_a_missing_data_directory_under_a_folder_this_account_can_write_to_is_information(
        self, tmp_path: Path
    ) -> None:
        [finding] = doctor.data_directory_findings(settings_for(tmp_path))

        assert finding.level is Level.INFO
        assert finding.title == f"The data directory {tmp_path / 'data'} does not exist yet; the first start makes it"

    def test_an_existing_data_directory_is_ok(self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine) -> None:
        (tmp_path / "data").mkdir(mode=0o700)

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

    def test_an_owner_only_data_directory_this_account_can_write_to_is_safe_to_use(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir(mode=0o700)

        assert titles(doctor.data_directory_findings(settings), Level.OK) == [
            f"The data directory {data} is safe to use"
        ]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can open a folder with no read bit")
    @pytest.mark.parametrize("mode", [0o300, 0o100, 0o000])
    def test_an_owner_only_data_directory_this_account_cannot_open_is_a_failure_that_cannot_be_examined(
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

        [failure] = findings
        assert failure.level is Level.FAIL
        assert failure.title == f"The data directory cannot be examined: cannot examine {data}: PermissionError"
        assert "is safe to use" not in " ".join(titles(findings))

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a folder with no write bit")
    @pytest.mark.parametrize("mode", [0o500, 0o400, 0o554, 0o550, 0o444, 0o505])
    def test_an_owner_only_data_directory_this_account_cannot_write_to_is_a_failure_not_safe_to_use(
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

        [failure] = findings
        assert failure.level is Level.FAIL
        assert failure.title == f"This account cannot write to the data directory {data}"
        assert failure.details == ("The database, its journal, and the access token live there, so collection fails.",)
        assert failure.fix == "Run `usagedash agent install`, which makes it writable by you alone again."
        assert "is safe to use" not in " ".join(titles(findings))

    def test_the_data_directory_is_asked_for_write_and_search_access_by_this_account(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir(mode=0o700)
        asked: list[tuple[Path, int]] = []

        def cannot_write(path: Path, mode: int) -> bool:
            asked.append((Path(path), mode))
            return False

        monkeypatch.setattr(doctor.os, "access", cannot_write)

        [failure] = doctor.data_directory_findings(settings)

        assert failure.level is Level.FAIL
        assert asked == [(data, os.W_OK | os.X_OK)]

    @pytest.mark.parametrize("mode", [0o750, 0o755, 0o705, 0o701, 0o710, 0o740, 0o704, 0o754])
    def test_a_data_directory_others_can_open_is_a_warning_that_is_not_safe_to_use(
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

        [warning] = findings
        assert warning.level is Level.WARN
        assert warning.title == (
            f"Other accounts can look at the data directory {data}: "
            f"its permissions ({mode:04o}) let other accounts open it"
        )
        assert warning.fix == "Run `usagedash agent install`, which makes it owner-only again."
        assert "is safe to use" not in " ".join(titles(findings))

    def test_a_data_directory_with_an_access_control_list_is_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        data = tmp_path / "data"
        data.mkdir(mode=0o700)
        monkeypatch.setattr(doctor, "what_lets_another_account_look", lambda _path: "it has an access-control list")

        [warning] = doctor.data_directory_findings(settings)

        assert warning.level is Level.WARN
        assert warning.title == f"Other accounts can look at the data directory {data}: it has an access-control list"

    def test_a_data_directory_that_cannot_be_examined_for_who_can_look_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").mkdir(mode=0o700)

        def unreadable(path: Path) -> str | None:
            raise ConfigError(f"cannot examine {path}: PermissionError")

        monkeypatch.setattr(doctor, "what_lets_another_account_look", unreadable)

        [failure] = doctor.data_directory_findings(settings)

        assert failure.level is Level.FAIL
        assert failure.title == (
            f"The data directory cannot be examined: cannot examine {tmp_path / 'data'}: PermissionError"
        )

    def test_a_data_directory_another_account_can_write_is_a_failure_and_is_not_also_asked_who_can_look(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = settings_for(tmp_path)
        (tmp_path / "data").mkdir(mode=0o700)
        monkeypatch.setattr(doctor, "what_lets_another_account_change", lambda _path: "it is writable")

        def forbidden(_path: Path) -> str | None:
            raise AssertionError("the look check must not run after a change problem")

        monkeypatch.setattr(doctor, "what_lets_another_account_look", forbidden)

        assert [finding.level for finding in doctor.data_directory_findings(settings)] == [Level.FAIL]

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
        (tmp_path / "data").mkdir(mode=0o700)
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

    def test_a_recorded_path_directory_the_collector_would_discard_does_not_supply_a_cli(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        directory = tmp_path / "agent-bin"
        executable_file(directory / "claude")
        executable_file(directory / "loose", mode=0o777)
        machine.environment = {"PATH": str(directory)}

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert only(findings, "claude: not found").level is Level.INFO

    def test_the_same_directory_without_the_unsafe_entry_supplies_the_cli(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        directory = tmp_path / "agent-bin"
        on_the_path = executable_file(directory / "claude")
        machine.environment = {"PATH": str(directory)}

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert only(findings, "claude: ").title == f"claude: {on_the_path.resolve()}"

    def test_a_path_with_every_directory_discarded_finds_nothing_and_never_searches_the_current_directory(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discarded = tmp_path / "discarded"
        executable_file(discarded / "loose", mode=0o777)
        executable_file(tmp_path / "claude")
        monkeypatch.chdir(tmp_path)
        machine.environment = {"PATH": str(discarded)}

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert only(findings, "claude: not found").level is Level.INFO

    def test_without_an_installed_agent_a_bare_name_is_looked_up_on_this_shells_path(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_shell_path = executable_file(tmp_path / "shell-bin" / "codex")
        monkeypatch.setenv("PATH", str(on_shell_path.parent))
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "codex: ").title == f"codex: {on_shell_path.resolve()}"

    def test_without_an_installed_agent_a_relative_path_entry_is_resolved_from_the_directory_doctor_runs_in(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        relative_to_cwd = executable_file(tmp_path / "relative-bin" / "codex")
        monkeypatch.setenv("PATH", "relative-bin")
        monkeypatch.chdir(tmp_path)
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "codex: ").title == f"codex: {relative_to_cwd.resolve()}"

    def test_a_relative_path_entry_finds_nothing_from_another_directory(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executable_file(tmp_path / "relative-bin" / "codex")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv("PATH", "relative-bin")
        monkeypatch.chdir(elsewhere)
        machine.environment, machine.state = None, None

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert only(findings, "codex: not found").level is Level.INFO

    @pytest.mark.parametrize(
        ("environment", "expected"),
        [
            (None, "SHELL-PATH"),
            ({}, "DEFAULT-PATH"),
            ({"USAGEDASH_PORT": "9123"}, "DEFAULT-PATH"),
            ({"PATH": "/recorded/bin"}, "/recorded/bin"),
            ({"PATH": ""}, ""),
        ],
    )
    def test_the_agents_path_is_its_recorded_one_then_the_default_and_only_when_nothing_is_installed_this_shells(
        self, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str] | None, expected: str
    ) -> None:
        monkeypatch.setenv("PATH", "SHELL-PATH")
        monkeypatch.setattr(doctor.os, "defpath", "DEFAULT-PATH")

        assert doctor._agent_path(environment) == expected

    def test_a_property_list_that_records_no_path_is_judged_by_the_default_path_not_this_shells(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_shell_path = executable_file(tmp_path / "shell-bin" / "codex")
        on_default_path = executable_file(tmp_path / "default-bin" / "claude")
        monkeypatch.setenv("PATH", str(on_shell_path.parent))
        monkeypatch.setattr(doctor.os, "defpath", str(on_default_path.parent))
        machine.environment = {"USAGEDASH_PORT": str(PORT)}

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert only(findings, "codex: not found").level is Level.INFO
        assert only(findings, "claude: ").title == f"claude: {on_default_path.resolve()}"

    def test_an_os_error_while_resolving_one_cli_is_a_warning_for_that_cli_and_the_others_are_still_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = doctor.checked_command

        def resolve(
            command: list[str], environment: dict[str, str], directory: Path, *, fallback_name: str | None = None
        ) -> list[str]:
            if command == ["claude"]:
                raise PermissionError("password=hunter2")
            return real(command, environment, directory, fallback_name=fallback_name)

        monkeypatch.setattr(doctor, "checked_command", resolve)

        findings = doctor.vendor_findings(settings_for(tmp_path), "")

        warning = only(findings, "claude: ")
        assert warning.level is Level.WARN
        assert warning.title == "claude: could not examine the executable: PermissionError"
        assert warning.fix == "Check the permissions of claude's folder, then run `usagedash doctor` again."
        assert "hunter2" not in repr(findings)
        assert only(findings, "codex: not found").level is Level.INFO
        assert only(findings, "grok: not found").level is Level.INFO

    @pytest.mark.parametrize(
        ("function", "what"),
        [
            ("system_findings", "system"),
            ("settings_findings", "settings"),
            ("data_directory_findings", "data directory"),
            ("vendor_findings", "command line tool"),
            ("path_findings", "PATH"),
            ("agent_helper_findings", "helper"),
            ("_answer_findings", "port"),
        ],
    )
    def test_a_check_that_raises_is_a_failure_naming_only_its_type_and_every_other_check_still_runs(
        self,
        tmp_path: Path,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        function: str,
        what: str,
    ) -> None:
        def explode(*_arguments: object, **_keywords: object) -> list[Finding]:
            raise RuntimeError("password=hunter2")

        machine.running()
        monkeypatch.setattr(doctor, function, explode)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        [failure] = [found for found in findings if f"The {what} check could not finish" in found.title]
        assert failure.level is Level.FAIL
        assert failure.title == f"The {what} check could not finish: RuntimeError"
        assert failure.details == ("Every other check ran.",)
        assert failure.fix == "Run `usagedash doctor` again, and if it ends the same way, post this report in an issue."
        assert "hunter2" not in repr(findings)
        assert len(findings) > 3
        assert "launchd is running the agent (pid 838)" in titles(findings)

    def test_a_log_check_that_raises_is_a_failure_too(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        machine.health = HealthResult(HealthKind.REFUSED)

        def explode(_paths: agent.AgentPaths) -> list[Finding]:
            raise RuntimeError("password=hunter2")

        monkeypatch.setattr(doctor, "log_findings", explode)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        assert "The log check could not finish: RuntimeError" in titles(findings, Level.FAIL)
        assert "hunter2" not in repr(findings)

    def test_the_checks_after_an_install_are_guarded_the_same_way(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*_arguments: object, **_keywords: object) -> list[Finding]:
            raise RuntimeError("password=hunter2")

        machine.running()
        monkeypatch.setattr(doctor, "_answer_findings", explode)
        monkeypatch.setattr(doctor, "log_findings", explode)

        findings = doctor.diagnose_agent(paths, PORT)

        assert titles(findings, Level.FAIL) == [
            "The port check could not finish: RuntimeError",
            "The log check could not finish: RuntimeError",
        ]
        assert "hunter2" not in repr(findings)

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
        assert finding.title == "The directory of a vendor CLI would be left off the agent's PATH: /opt/tools/bin"
        assert finding.details == (
            "its permissions (0775) let other accounts write",
            "The CLI starts from its recorded path, but anything it starts by name from that directory (a "
            "Node-based CLI looks for `node`) will not be found.",
        )
        assert finding.fix is not None
        assert "writable by you alone" in finding.fix

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
        log.write_text("word " * 300 + "\n")

        shown = doctor.tail_lines(log)

        assert shown == [("word " * 300)[: doctor.MAX_LOG_LINE_CHARACTERS] + "…"]

    def test_a_last_line_longer_than_the_window_is_reported_not_silently_dropped(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("x" * 50)

        assert doctor.tail_lines(log, max_bytes=10) == [
            f"(the last line is longer than 10 bytes and is not shown; open {log} to read it)"
        ]

    def test_the_cut_end_of_an_oversized_last_line_is_never_shown_because_its_credential_name_may_be_cut_off(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "log"
        log.write_text("password=" + "hunter2" * 20)

        shown = doctor.tail_lines(log, max_bytes=40)

        assert shown is not None
        assert "hunter2" not in "".join(shown)
        assert len(shown) == 1

    def test_a_cut_first_line_is_dropped_when_whole_lines_follow_it(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("older\nnewer\n")

        assert doctor.tail_lines(log, max_bytes=8) == ["newer"]

    def test_a_last_line_exactly_as_long_as_the_window_is_shown_whole(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("x" * 10)

        assert doctor.tail_lines(log, max_bytes=10) == ["x" * 10]

    @pytest.mark.parametrize(
        "separator",
        ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "\r", "\r\r", "\v\f"],
    )
    def test_a_character_that_splitlines_treats_as_a_line_end_does_not_part_a_credential_from_its_name(
        self, tmp_path: Path, separator: str
    ) -> None:
        log = tmp_path / "log"
        log.write_bytes(f"password={separator}hunter2\nafter\n".encode())

        shown = doctor.tail_lines(log)

        assert shown is not None
        assert "hunter2" not in "".join(shown)
        assert shown[-1] == "after"
        assert len(shown) == 2

    def test_a_windows_line_ending_does_not_leave_a_carriage_return_on_the_line(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_bytes(b"first\r\nsecond\r\n")

        assert doctor.tail_lines(log) == ["first", "second"]

    def test_blank_lines_are_kept_so_the_log_keeps_its_shape(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("first\n\nthird\n")

        assert doctor.tail_lines(log) == ["first", "", "third"]

    def test_a_log_that_does_not_end_with_a_line_feed_keeps_its_last_line(self, tmp_path: Path) -> None:
        log = tmp_path / "log"
        log.write_text("first\nsecond")

        assert doctor.tail_lines(log) == ["first", "second"]

    def test_a_credential_or_an_email_address_in_a_log_line_is_replaced_before_it_is_shown(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "log"
        log.write_text("owner@example.com used api_key=sk-test-value on /api/state\nAccepted GET /api/state: ok\n")

        assert doctor.tail_lines(log) == [
            "<email> used api_key=<redacted>",
            "Accepted GET /api/state: ok",
        ]

    @pytest.mark.parametrize(
        ("line", "screened"),
        [
            ("Accepted GET /api/state: signature", "Accepted GET /api/state: signature"),
            ("cannot read access-token: PermissionError", "cannot read access-token: <redacted>"),
            (
                "/Users/greg/.venv/lib/python3.13/site-packages/usagedash/cli.py",
                "/Users/greg/.venv/lib/python3.13/site-packages/usagedash/cli.py",
            ),
            ("owner@example.com failed", "<email> failed"),
            ("first.last+tag@mail.example.co.uk failed", "<email> failed"),
            ("Authorization: Bearer abcdef", "Authorization: <redacted>"),
            ("Authorization: Basic dXNlcjpwYXNz", "Authorization: <redacted>"),
            ("sent bearer abcdef now", "sent Bearer <redacted> now"),
            ("api_key=sk-test-value", "api_key=<redacted>"),
            ("API-KEY: sk-test-value", "API-KEY: <redacted>"),
            ("password: hunter2", "password: <redacted>"),
            ("passwd = hunter2", "passwd = <redacted>"),
            ("secret=abc", "secret=<redacted>"),
            ("a" * 32, "<redacted>"),
            ("a" * 31, "a" * 31),
            ("sha " + "0123456789abcdef" * 4, "sha <redacted>"),
            ("owner@example.com api_key=abc", "<email> api_key=<redacted>"),
            ('{"password": "abc123"}', '{"password": <redacted>'),
            ("{'api_key': 'abc'}", "{'api_key': <redacted>"),
            ("DB_PASSWORD=abc123", "DB_PASSWORD=<redacted>"),
            ("OPENAI_API_KEY=sk-abc", "OPENAI_API_KEY=<redacted>"),
            ("x-api-key: abc", "x-api-key: <redacted>"),
            ('client_secret = "abc def"', "client_secret = <redacted>"),
            ('password="abc",next=1', "password=<redacted>"),
            ('{"user": "greg", "password": "p w", "ok": 1}', '{"user": "greg", "password": <redacted>'),
            ("", ""),
        ],
    )
    def test_screening_replaces_what_could_be_a_credential_and_nothing_else(self, line: str, screened: str) -> None:
        assert doctor.screen_for_sharing(line) == screened

    @pytest.mark.parametrize(
        "label",
        [
            "API Key",
            "api key",
            "Api-Key",
            "apikey",
            "API_KEY",
            "Access Key",
            "access-key",
            "accesskey",
            "Private Key",
            "private_key",
            "privatekey",
            "Pass Word",
            "pass word",
            "PassWord",
            "Passphrase",
            "Pass Phrase",
            "pass_phrase",
            "pass-phrase",
            "passwd",
            "Pass Wd",
            "password",
        ],
    )
    @pytest.mark.parametrize("separator", [": ", "=", " = ", ":\t"])
    def test_a_label_spelled_with_a_space_a_hyphen_an_underscore_or_nothing_is_matched(
        self, label: str, separator: str
    ) -> None:
        assert doctor.screen_for_sharing(f"{label}{separator}short123") == f"{label}{separator}<redacted>"

    @pytest.mark.parametrize(
        "template",
        [
            "password={s}",
            'password="{s}"',
            "password='{s}'",
            '{{"password": "{s}"}}',
            'password="a\\"{s}"',
            "password='a\\'{s}'",
            "password=a,{s}",
            "password=a}}{s}",
            'password="a b {s}"',
            "Authorization: Basic {s}",
            "Authorization: Bearer {s}",
            "Authorization: Token abc {s}",
            "Authorization: Digest username=x, response={s}",
            "DB_PASSWORD = {s}",
            "x-api-key:{s}",
            'client_secret="p, q}} {s}"',
            "PASSWORD:\t{s}",
            '"secret" : "{s}"',
            "api-key = 'one' 'two' {s}",
            "passwd={s} trailing words",
            "password[0]={s}",
            "password.value={s}",
            'config["password"][0] = "{s}"',
            "secret ['a'] : {s}",
            "db.passwd (prod) = {s}",
            "AUTHORIZATION_HEADER\t=\t{s}",
            "session_token={s}",
            "auth.token: {s}",
            '{{"token": "{s}"}}',
            "Cookie: sessionid={s}",
            "Set-Cookie: a=b; {s}",
            "credential = {s}",
            "GOOGLE_APPLICATION_CREDENTIALS: {s}",
            "private_key: {s}",
            "token: PermissionError: {s}",
        ],
    )
    def test_no_value_syntax_lets_a_secret_survive_the_screen_after_a_credential_name(self, template: str) -> None:
        secret = "S3cr3tValue"

        screened = doctor.screen_for_sharing(template.format(s=secret))

        assert secret not in screened
        assert doctor.REDACTED in screened

    @pytest.mark.parametrize(
        "text",
        [
            "password:\nthe next line",
            "password=\n       -> the remedy",
            "Bearer\nResult: all 1 checks passed.",
            "bearer \nResult: all 1 checks passed.",
            "api_key :\n\n\nthe line after two blank ones",
            "secret\t:\t\nafter a tab",
            "first\npassword:\nsecond\nBearer\nResult: 1 problem found.",
            "authorization:\r\nafter a carriage return",
            "\n\npassword:\n\n",
            "",
        ],
    )
    def test_screening_never_merges_removes_or_adds_a_line(self, text: str) -> None:
        screened = doctor.screen_for_sharing(text)

        before, after = text.split("\n"), screened.split("\n")
        assert len(after) == len(before)
        for original, kept in zip(before, after, strict=True):
            if not re.search(r"(?i)password|secret|api_key|authorization|bearer", original):
                assert kept == original

    def test_a_trailing_credential_name_does_not_take_the_verdict_or_the_remedy_after_it(self, tmp_path: Path) -> None:
        findings = [
            Finding(Level.INFO, "Last lines of the log", ("2026-11-01 INFO something: password:",)),
            Finding(Level.FAIL, "something failed", ("Authorization:",), "Run `usagedash agent install` again."),
            Finding(Level.WARN, "a trailing word", ("it ends in Bearer",)),
        ]

        report = doctor.render(findings, tmp_path)

        assert "-> Run `usagedash agent install` again." in report
        assert "[WARN] a trailing word" in report
        assert [line for line in report.splitlines() if line.startswith("Result:")] == [
            "Result: 1 problem found. Fix the first FAIL above, then run `usagedash doctor` again."
        ]

    @pytest.mark.parametrize(
        ("line", "screened"),
        [
            ("https://user:hunter2@localhost:8787/path", "https://<redacted>@localhost:8787/path"),
            ("postgres://admin:S3cr3t@db.internal/app", "postgres://<redacted>@db.internal/app"),
            ("postgres://user@example.com:hunter2@localhost/db", "postgres://<redacted>@localhost/db"),
            ("https://user:hunt@er2@localhost/path", "https://<redacted>@localhost/path"),
            ("http://a:b@c@d:e@f/x", "http://<redacted>@f/x"),
            ("https://user:pass@host?next=a@b", "https://<redacted>@host?next=a@b"),
            ("https://user:pass@host#frag@x", "https://<redacted>@host#frag@x"),
            (
                "log https://u:p@ss@host/ and https://v:q@host/",
                "log https://<redacted>@host/ and https://<redacted>@host/",
            ),
            ("https://user@localhost/path", "https://user@localhost/path"),
            ("https://user@host:8080/path", "https://user@host:8080/path"),
            ("ssh://git@github.com:22/org/repo.git", "ssh://<email>:22/org/repo.git"),
            ("https://host/path@with:colons@x", "https://host/path@with:colons@x"),
            ("see https://example.com/path:with:colons", "see https://example.com/path:with:colons"),
        ],
    )
    def test_the_user_and_password_in_a_url_are_replaced_and_an_ordinary_url_is_not(
        self, line: str, screened: str
    ) -> None:
        assert doctor.screen_for_sharing(line) == screened

    @pytest.mark.parametrize(
        "value",
        [
            "ValidError",
            "PermissionError",
            "FileNotFoundError",
            "usagedash.errors.ConfigError",
            "SomeOtherException",
            "S3cr3tValueError",
        ],
    )
    def test_a_value_that_looks_like_an_exception_type_is_replaced_like_any_other_value(self, value: str) -> None:
        assert doctor.screen_for_sharing(f"password: {value}") == "password: <redacted>"
        assert doctor.screen_for_sharing(f"cannot read access-token: {value}") == "cannot read access-token: <redacted>"

    @pytest.mark.parametrize("name_length", [0, 1, 63, 64, 65, 66, 200, 1000])
    def test_a_credential_name_is_matched_however_long_the_text_before_its_separator_is(self, name_length: int) -> None:
        name = "password" + ".part" * (name_length // 5) + "x" * (name_length % 5)

        assert doctor.screen_for_sharing(f"{name}=hunter2") == f"{name}=<redacted>"

    def test_a_long_dotted_credential_field_name_does_not_let_a_short_value_through(self) -> None:
        line = "password" + ".part" * 14 + "=hunter2"

        assert "hunter2" not in doctor.screen_for_sharing(line)

    def test_a_line_with_many_credential_words_and_no_separator_is_left_alone(self) -> None:
        line = "token " * 2000

        assert doctor.screen_for_sharing(line) == line

    def test_a_sentence_that_only_mentions_a_credential_name_far_from_any_separator_is_left_alone(self) -> None:
        line = "the password field is described in the long help text that follows and ends here without a separator"

        assert doctor.screen_for_sharing(line) == line

    @pytest.mark.parametrize("field", ["title", "detail", "fix"])
    def test_a_newline_in_a_field_cannot_add_a_line_to_the_report_such_as_a_forged_verdict(
        self, tmp_path: Path, field: str
    ) -> None:
        forged = "x\nResult: all 99 checks passed. Run `usagedash open` to view the dashboard."
        finding = Finding(
            Level.FAIL,
            forged if field == "title" else "real title",
            (forged,) if field == "detail" else (),
            forged if field == "fix" else "real fix",
        )

        report = doctor.render([finding], tmp_path)

        verdicts = [line for line in report.splitlines() if line.startswith("Result:")]
        assert verdicts == ["Result: 1 problem found. Fix the first FAIL above, then run `usagedash doctor` again."]
        assert "x?Result: all 99 checks passed" in report

    def test_an_escape_byte_and_other_control_characters_in_a_field_are_replaced_in_the_report(
        self, tmp_path: Path
    ) -> None:
        finding = Finding(Level.WARN, "title\x1b[2J\x07\x00 end", ("detail\r\x7f\x9b",), None)

        report = doctor.render([finding], tmp_path)

        assert not [character for character in report if not character.isprintable() and character != "\n"]
        assert "title?[2J?? end" in report
        assert "detail???" in report

    def test_a_data_directory_path_with_a_newline_in_it_cannot_forge_a_verdict_line(self, tmp_path: Path) -> None:
        settings = load_settings(
            {
                "USAGEDASH_DATA_DIR": str(tmp_path / "x\nResult: all 9 checks passed. Run `usagedash open` now."),
                "USAGEDASH_PORT": str(PORT),
            },
            home=tmp_path,
        )

        report = doctor.render(doctor.settings_findings(settings, None), tmp_path)

        verdicts = [line for line in report.splitlines() if line.startswith("Result:")]
        assert verdicts == ["Result: all 1 checks passed. Run `usagedash open` to view the dashboard."]

    def test_the_whole_report_is_screened_not_only_the_log_lines(self, tmp_path: Path) -> None:
        finding = Finding(
            Level.WARN,
            f"claude: cannot use {tmp_path}/owner@example.com/bin",
            ("port must be an integer, got 'password=hunter2'",),
        )

        report = doctor.render([finding], tmp_path)

        assert "owner@example.com" not in report
        assert "hunter2" not in report
        assert "<email>" in report
        assert "password=<redacted>" in report

    def test_the_failure_report_after_an_install_is_screened_too(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine
    ) -> None:
        machine.health = HealthResult(HealthKind.ANSWERED, 404)
        machine.holder = PortHolder(77, "owner@example.com")

        report = doctor.render_agent_failure(paths, PORT, tmp_path)

        assert "owner@example.com" not in report
        assert f"Port {PORT} is held by <email> (pid 77)." in report

    def test_a_settings_error_that_echoes_an_email_address_does_not_reach_the_pasted_output(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.nothing_installed()
        monkeypatch.setenv("USAGEDASH_PORT", "owner@example.com")

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "owner@example.com" not in output
        assert "port must be an integer, got '<email>'" in output


@pytest.mark.skipif(sys.platform != "darwin", reason="the required helper is prepared on macOS only")
class TestTheInstalledAgentsHelper:
    """What the installed agent's collector will meet preparing `security`, asked with its own call and settings."""

    @pytest.fixture(autouse=True)
    def helper(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        helper = executable_file(tmp_path / "usr-bin" / "security")
        monkeypatch.setattr(executables, "REQUIRED_PATH_HELPERS", (helper,))
        return helper

    @staticmethod
    def served_in(tmp_path: Path, name: str = "served-data") -> Settings:
        return load_settings({"USAGEDASH_DATA_DIR": str(tmp_path / name), "USAGEDASH_PORT": str(PORT)}, home=tmp_path)

    @staticmethod
    def path_without_the_helper(tmp_path: Path) -> str:
        directory = tmp_path / "path"
        directory.mkdir(mode=0o700)
        return str(directory)

    def test_a_helper_that_can_be_prepared_is_no_finding_and_nothing_is_created(self, tmp_path: Path) -> None:
        served = self.served_in(tmp_path)

        findings = doctor.agent_helper_findings(served, self.path_without_the_helper(tmp_path))

        assert findings == []
        assert not served.data_dir.exists()

    def test_the_environment_the_collector_builds_is_judged_without_being_changed(self, tmp_path: Path) -> None:
        data = tmp_path / "served-data"

        environment = executables.execution_environment(
            {"PATH": self.path_without_the_helper(tmp_path)},
            Path.cwd(),
            helper_directory=data,
            change_nothing=True,
        )

        assert environment["PATH"].endswith(str(data.resolve() / "agent-bin"))
        assert not data.exists()

    def test_an_unsafe_entry_in_the_installed_agents_own_folder_is_a_warning_naming_it(
        self, tmp_path: Path, helper: Path
    ) -> None:
        served = self.served_in(tmp_path)
        folder = served.data_dir / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(helper.resolve())
        sibling = executable_file(folder / "loose", mode=0o777)

        [finding] = doctor.agent_helper_findings(served, self.path_without_the_helper(tmp_path))

        assert finding.level is Level.WARN
        assert finding.title.startswith("The installed agent cannot prepare a helper its command line tools need: ")
        assert str(sibling) in finding.title
        assert finding.details == (
            "The collector refuses to start a tool with this error, so its accounts show no data.",
        )
        assert finding.fix == "Fix what that names, then run `usagedash agent install`."

    def test_a_link_that_names_another_target_is_a_warning(self, tmp_path: Path) -> None:
        served = self.served_in(tmp_path)
        folder = served.data_dir / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        (folder / "security").symlink_to(tmp_path / "somewhere-else")

        [finding] = doctor.agent_helper_findings(served, self.path_without_the_helper(tmp_path))

        assert finding.title.endswith("it names an unexpected target")

    def test_a_helper_already_on_the_recorded_path_needs_no_folder_and_so_no_finding(
        self, tmp_path: Path, helper: Path
    ) -> None:
        served = self.served_in(tmp_path)
        folder = served.data_dir / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        executable_file(folder / "loose", mode=0o777)

        assert doctor.agent_helper_findings(served, str(helper.parent)) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a folder with no write bit")
    def test_a_folder_the_installed_agent_cannot_write_a_link_into_is_a_warning(self, tmp_path: Path) -> None:
        served = self.served_in(tmp_path)
        folder = served.data_dir / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        folder.chmod(0o500)
        try:
            [finding] = doctor.agent_helper_findings(served, self.path_without_the_helper(tmp_path))
        finally:
            folder.chmod(0o700)

        assert finding.title.endswith(f"cannot prepare required helpers in {folder.resolve()}: PermissionError")

    def test_the_installed_agents_data_directory_and_path_are_asked_not_this_shells(
        self,
        tmp_path: Path,
        paths: agent.AgentPaths,
        machine: Machine,
        helper: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(executables, "REQUIRED_PATH_HELPERS", (helper,))
        served = self.served_in(tmp_path, "served-data")
        shell = self.served_in(tmp_path, "shell-data")
        folder = served.data_dir / "agent-bin"
        folder.mkdir(parents=True, mode=0o700)
        executable_file(folder / "loose", mode=0o777)
        machine.environment = {"PATH": self.path_without_the_helper(tmp_path)}

        findings = doctor.diagnose(paths, shell=shell, served=served)

        [finding] = [found for found in findings if "cannot prepare a helper" in found.title]
        assert str(served.data_dir) in finding.title
        assert str(shell.data_dir) not in finding.title

    def test_nothing_is_asked_when_no_agent_is_installed(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*_arguments: object, **_keywords: object) -> list[Finding]:
            raise AssertionError("the installed agent's helper must not be asked when nothing is installed")

        machine.nothing_installed()
        monkeypatch.setattr(doctor, "agent_helper_findings", forbidden)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert not [title for title in titles(findings) if "helper" in title]


class TestAnUnexpectedErrorNeverEndsTheReport:
    @pytest.mark.parametrize("function", ["installed_environment", "service_state"])
    def test_an_error_asking_about_the_agent_is_a_failure_naming_its_type_and_the_other_checks_still_run(
        self,
        tmp_path: Path,
        paths: agent.AgentPaths,
        machine: Machine,
        monkeypatch: pytest.MonkeyPatch,
        function: str,
    ) -> None:
        def explode(_paths: agent.AgentPaths) -> object:
            raise RuntimeError("password=hunter2")

        machine.running()
        monkeypatch.setattr(agent, function, explode)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=settings_for(tmp_path))

        [failure] = [found for found in findings if "The agent check could not finish" in found.title]
        assert failure.level is Level.FAIL
        assert failure.title == "The agent check could not finish: RuntimeError"
        assert failure.details == ("Every other check ran.",)
        assert "hunter2" not in repr(findings)
        assert any(title.startswith("Settings load") for title in titles(findings))
        assert any("data directory" in title for title in titles(findings))

    def test_a_failed_agent_check_is_unknown_and_not_nothing_installed_so_this_shells_settings_do_not_stand_in(
        self, tmp_path: Path, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_paths: agent.AgentPaths) -> object:
            raise RuntimeError("synthetic")

        monkeypatch.setattr(agent, "service_state", explode)

        findings = doctor.diagnose(paths, shell=settings_for(tmp_path), served=None)

        assert titles(findings, Level.FAIL) == ["The agent check could not finish: RuntimeError"]
        assert not [title for title in titles(findings) if title.startswith(("claude:", "codex:", "grok:"))]
        assert not [title for title in titles(findings) if "data directory" in title]

    def test_the_checks_after_an_install_survive_an_error_asking_about_the_agent(
        self, paths: agent.AgentPaths, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_paths: agent.AgentPaths) -> object:
            raise RuntimeError("password=hunter2")

        monkeypatch.setattr(agent, "service_state", explode)

        findings = doctor.diagnose_agent(paths, PORT)

        assert "The agent check could not finish: RuntimeError" in titles(findings, Level.FAIL)
        assert "hunter2" not in repr(findings)

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("password=hunter2"),
            OSError("input/output error"),
            PermissionError("denied"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
            ValueError("password=hunter2"),
        ],
    )
    def test_a_settings_loader_that_raises_something_unexpected_is_a_config_error_naming_only_its_type(
        self, error: Exception
    ) -> None:
        def load() -> Settings:
            raise error

        outcome = cli._settings_outcome(load)

        assert isinstance(outcome, ConfigError)
        assert str(outcome) == f"the settings could not be read: {type(error).__name__}"

    def test_an_error_this_package_raised_is_returned_as_it_is(self) -> None:
        raised = ConfigError("port must be an integer, got 'x'")

        def load() -> Settings:
            raise raised

        assert cli._settings_outcome(load) is raised

    def test_a_loader_that_returns_nothing_is_nothing(self) -> None:
        assert cli._settings_outcome(lambda: None) is None

    def test_an_interrupt_is_not_swallowed_by_the_settings_boundary(self) -> None:
        def load() -> Settings:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            cli._settings_outcome(load)

    def test_the_doctor_command_still_prints_its_report_when_the_settings_cannot_be_read(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.nothing_installed()

        def unreadable() -> Settings:
            raise OSError("password=hunter2")

        monkeypatch.setattr(cli, "_shell_settings_as_install_sees_them", unreadable)

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "the settings could not be read: OSError" in output
        assert "hunter2" not in output
        assert "Result:" in output


class TestFindingWhoHoldsAPort:
    def test_lsof_is_asked_for_the_listener_and_its_process_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = FakeRun(stdout="p838\ncpython3.13\nf12\nn127.0.0.1:8787\n")
        monkeypatch.setattr(doctor.subprocess, "run", run)

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")
        assert run.commands == [[doctor.LSOF_BINARY, "-nP", "-iTCP:8787", "-sTCP:LISTEN", "-Fpcn"]]

    def test_two_processes_that_both_listen_where_the_loopback_address_reaches_are_no_holder_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = "p838\ncpython3.13\nf12\nn127.0.0.1:8787\np900\ncother\nf3\nn*:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) is None

    def test_one_process_listening_twice_is_still_one_holder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = "p838\ncpython3.13\nf12\nn127.0.0.1:8787\nf13\nn*:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")

    def test_a_second_process_on_another_interface_does_not_make_the_holder_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = "p838\ncpython3.13\nf12\nn127.0.0.1:8787\np900\ncother\nf3\nn192.168.1.5:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")

    def test_a_socket_that_cannot_be_created_is_a_port_that_is_not_free_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_arguments: object, **_keywords: object) -> socket.socket:
            raise OSError("Too many open files")

        monkeypatch.setattr(doctor.socket, "socket", refuse)

        assert doctor._port_is_free(8788) is False
        assert doctor.suggest_port(8787) is None

    @pytest.mark.parametrize(
        ("listener", "found"),
        [
            ("127.0.0.1:8787", True),
            ("*:8787", True),
            ("192.168.1.5:8787", False),
            ("10.0.0.2:8787", False),
            ("[::1]:8787", False),
            ("[::]:8787", False),
            ("localhost:8787", False),
            ("8787", False),
            ("", False),
        ],
    )
    def test_only_a_listener_the_loopback_address_reaches_counts_as_the_holder(
        self, monkeypatch: pytest.MonkeyPatch, listener: str, found: bool
    ) -> None:
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=f"p838\ncpython3.13\nf12\nn{listener}\n"))

        assert (doctor.port_holder(8787) == PortHolder(838, "python3.13")) is found

    def test_a_listener_on_another_interface_does_not_hide_the_one_that_answers_the_loopback_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = "p7\ncelsewhere\nf4\nn192.168.1.5:8787\np838\ncpython3.13\nf12\nn127.0.0.1:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")

    def test_one_process_with_a_listener_elsewhere_and_one_on_loopback_is_the_holder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = "p838\ncpython3.13\nf4\nn192.168.1.5:8787\nf5\nn127.0.0.1:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) == PortHolder(838, "python3.13")

    @pytest.mark.parametrize(
        "output",
        [
            "",
            "pabc\ncfoo\nn127.0.0.1:8787\n",
            "p12\n",
            "cfoo\n",
            "garbage",
            "n127.0.0.1:8787\n",
            "p-4\nn127.0.0.1:8787\n",
        ],
    )
    def test_output_that_does_not_name_a_process_means_none_is_found(
        self, monkeypatch: pytest.MonkeyPatch, output: str
    ) -> None:
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(returncode=1, stdout=output))

        assert doctor.port_holder(8787) is None

    @pytest.mark.parametrize("returncode", [1, 2, 127, -9])
    def test_a_failed_lsof_names_no_holder_whatever_it_printed_before_it_failed(
        self, monkeypatch: pytest.MonkeyPatch, returncode: int
    ) -> None:
        output = "p838\ncpython3.13\nf12\nn127.0.0.1:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(returncode=returncode, stdout=output))

        assert doctor.port_holder(8787) is None

    def test_a_missing_or_stuck_lsof_means_none_is_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for failure in (FileNotFoundError("lsof"), doctor.subprocess.TimeoutExpired(["lsof"], 10)):
            monkeypatch.setattr(doctor.subprocess, "run", FakeRun(error=failure))
            assert doctor.port_holder(8787) is None

    @pytest.mark.parametrize("separator", ["\v", "\f", "\x85", "\u2028", "\r"])
    def test_a_process_name_holding_a_line_separator_cannot_forge_a_listener_field(
        self, monkeypatch: pytest.MonkeyPatch, separator: str
    ) -> None:
        output = f"p838\ncpython{separator}n127.0.0.1:8787\nf12\nn192.168.1.5:8787\n"
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout=output))

        assert doctor.port_holder(8787) is None

    def test_a_command_name_is_made_printable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.subprocess, "run", FakeRun(stdout="p5\nca\x1bb\nf3\nn127.0.0.1:8787\n"))

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
        machine.holder = PortHolder(838, "python3.13")

        assert cli.main(["doctor"]) == cli.EXIT_OK

        output = capsys.readouterr().out
        assert "Result: all" in output
        assert "FAIL" not in output
        assert not (home / "data").exists()

    def test_the_config_directory_is_guarded_as_agent_install_guards_it(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {}
        guarded: list[Path] = []

        def refuse(config_file: Path) -> None:
            guarded.append(config_file)
            raise ConfigError(f"cannot use {config_file.parent}: another account could replace it")

        monkeypatch.setattr(cli.standing, "_refuse_replaceable_config_directory", refuse)

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out.replace("~", str(home))
        assert guarded == [home / "data" / "config.toml", agents_config_file(home)]
        assert f"[FAIL] The settings cannot be loaded: cannot use {home / 'data'}: another account" in output
        assert "[FAIL] The installed agent's settings cannot be read: cannot use" in output
        assert "Settings load:" not in output

    def test_a_config_directory_that_passes_the_guard_loads_the_settings(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {}
        guarded: list[Path] = []
        monkeypatch.setattr(cli.standing, "_refuse_replaceable_config_directory", guarded.append)

        cli.main(["doctor"])

        assert guarded == [home / "data" / "config.toml", agents_config_file(home)]
        assert "[ok  ] Settings load:" in capsys.readouterr().out

    def test_the_installed_agents_config_directory_is_guarded_as_serve_guards_it(
        self, home: Path, machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        machine.environment = {}
        machine.holder = PortHolder(838, "python3.13")
        agents_config = agents_config_file(home)

        def refuse_only_the_agents(config_file: Path) -> None:
            if config_file == agents_config:
                raise ConfigError(f"cannot use {config_file.parent}: its permissions (0777) let other accounts write")

        monkeypatch.setattr(cli.standing, "_refuse_replaceable_config_directory", refuse_only_the_agents)

        assert cli.main(["doctor"]) == cli.EXIT_FAILURE

        output = capsys.readouterr().out
        assert "[ok  ] Settings load:" in output
        assert "[FAIL] The installed agent's settings cannot be read: cannot use" in output
        assert "its permissions (0777) let other accounts write" in output
        assert doctor._CORRECT_THE_SETTING in output
        assert machine.probed == []

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
        asked = self.answers(monkeypatch, False)

        assert self.wait(paths, clock) is False
        assert clock.now == cli.INSTALL_READY_WAIT_SECONDS
        assert len(asked) == int(cli.INSTALL_READY_WAIT_SECONDS / cli.INSTALL_READY_POLL_SECONDS)

    def test_a_probe_that_would_succeed_exactly_at_the_deadline_is_never_made(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        polls = int(cli.INSTALL_READY_WAIT_SECONDS / cli.INSTALL_READY_POLL_SECONDS)
        asked = self.answers(monkeypatch, *([False] * polls), True)

        assert self.wait(paths, clock) is False
        assert len(asked) == polls

    def test_no_probe_starts_after_the_deadline_when_the_launchd_question_is_what_crossed_it(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asked = self.answers(monkeypatch, False)

        def slow_question(_paths: agent.AgentPaths) -> agent.ServiceState | None:
            clock.now += cli.INSTALL_READY_WAIT_SECONDS + 5
            return RUNNING

        monkeypatch.setattr(agent, "service_state", slow_question)

        assert self.wait(paths, clock) is False
        assert len(asked) == 1
        assert clock.sleeps == []

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

    def test_a_listing_that_does_not_say_whether_the_agent_runs_does_not_end_the_wait_even_after_a_failed_run(
        self,
        paths: agent.AgentPaths,
        machine: Machine,
        clock: "TestWaitingForTheNewAgent.Clock",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        machine.state = agent.ServiceState("running", None, 2, "1", None)
        self.answers(monkeypatch, False, True)

        assert self.wait(paths, clock) is True
        assert clock.sleeps == [cli.INSTALL_READY_POLL_SECONDS]

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

    def test_an_agent_launchd_does_not_list_is_one_line(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(agent, "service_state", lambda _paths: None)

        assert cli.main(["agent", "status"]) == cli.EXIT_OK

        assert capsys.readouterr().out == f"{label}: not loaded\n"

    def test_status_asks_launchd_once_so_a_stopping_service_cannot_read_as_loaded_and_not_running(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        asked: list[agent.AgentPaths] = []

        def one_snapshot(paths: agent.AgentPaths) -> agent.ServiceState | None:
            asked.append(paths)
            return None

        def never(_paths: agent.AgentPaths) -> bool:
            raise AssertionError("a second question leaves a gap in which the answer can change")

        monkeypatch.setattr(agent, "service_state", one_snapshot)
        monkeypatch.setattr(agent, "is_loaded", never)

        cli.main(["agent", "status"])

        assert len(asked) == 1
        assert capsys.readouterr().out == f"{label}: not loaded\n"

    def test_a_running_agent_shows_its_pid_runs_logs_and_the_doctor_command(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
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
        monkeypatch.setattr(agent, "service_state", lambda _paths: CRASHED)

        assert cli.main(["agent", "status"]) == cli.EXIT_OK

        first = capsys.readouterr().out.splitlines()[0]
        assert first == f"{label}: loaded, not running; the last run ended with exit code 1"
        assert "running (pid" not in first

    def test_a_loaded_agent_killed_by_a_signal_is_reported_by_the_signal_not_an_exit_code(
        self, home: Path, label: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
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
        monkeypatch.setattr(
            agent, "service_state", lambda _paths: agent.ServiceState("waiting", None, None, None, None)
        )

        cli.main(["agent", "status"])

        assert capsys.readouterr().out.splitlines()[0] == f"{label}: loaded, not running (waiting)"

    @pytest.mark.parametrize(
        "state",
        [
            agent.ServiceState(None, None, None, None, None),
            agent.ServiceState("running", None, 2, "(never exited)", None),
            agent.ServiceState("running", None, 2, "1", None),
        ],
        ids=["nothing stated", "says running without a pid", "says running, last run failed"],
    )
    def test_a_loaded_agent_whose_listing_does_not_say_is_neither_called_running_nor_not_running(
        self,
        home: Path,
        label: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        state: agent.ServiceState,
    ) -> None:
        monkeypatch.setattr(agent, "service_state", lambda _paths: state)

        cli.main(["agent", "status"])

        first = capsys.readouterr().out.splitlines()[0]
        assert first == f"{label}: loaded; launchd's listing does not say whether it is running"
        assert "not running" not in first
        assert "running (pid" not in first
