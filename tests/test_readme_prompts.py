"""The prompts in the README are instructions a coding agent runs as the user. Their safety rules are locked here.

A prompt is code in everything but name: a missing check ships as a behavior. Each assertion states a rule that a
review found missing once, so editing the prose cannot quietly remove it.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"
REPOSITORY_ADDRESS = "https://github.com/chrisbaker2000/ai-usage-dashboard.git"


def prompts() -> dict[str, str]:
    """Every fenced ``text`` block in the README, keyed by its first line."""
    blocks = re.findall(r"```text\n(.*?)```", README.read_text(), flags=re.DOTALL)
    return {block.splitlines()[0]: block for block in blocks}


def prompt_starting(start: str) -> str:
    matches = [block for first_line, block in prompts().items() if first_line.startswith(start)]
    assert len(matches) == 1, f"expected exactly one prompt starting {start!r}, found {len(matches)}"
    return matches[0]


PROMPT_STARTS = {
    "install": "Install the AI usage dashboard",
    "repair": "The AI usage dashboard in ~/Developer/ai-usage-dashboard is not working",
    "account": "Help me add another account",
    "update": "Update the AI usage dashboard",
    "remove": "Remove the AI usage dashboard",
}


FORBIDDEN_TOOLS = (
    "Do not use sudo, edit the launchd property list by hand, run launchctl yourself, or change the repository's code."
)
"""The exact sentence, because ``"sudo" in text`` is also true of a prompt that says to use it."""
RESTRICTION_ON_TOOLS = {
    "install": FORBIDDEN_TOOLS,
    "repair": FORBIDDEN_TOOLS,
    "account": FORBIDDEN_TOOLS,
    "update": "do not use sudo or change the repository's code",
    "remove": "Do not use sudo or run launchctl yourself.",
}
CREDENTIALS_RULE = "Never read, print, copy, or send credentials"


class TestEveryPromptIsPresentAndForbidsTheDangerousThings:
    @pytest.mark.parametrize("name", PROMPT_STARTS)
    def test_the_prompt_exists_exactly_once(self, name: str) -> None:
        assert prompt_starting(PROMPT_STARTS[name])

    @pytest.mark.parametrize("name", RESTRICTION_ON_TOOLS)
    def test_the_prompt_forbids_sudo_and_hand_run_launchctl_in_so_many_words(self, name: str) -> None:
        text = prompt_starting(PROMPT_STARTS[name])

        assert RESTRICTION_ON_TOOLS[name] in text
        assert not re.search(r"(?<!not )\buse sudo\b", text, flags=re.IGNORECASE)

    @pytest.mark.parametrize("name", ["install", "repair", "account", "update"])
    def test_no_prompt_that_touches_credentials_lets_the_agent_read_or_send_them(self, name: str) -> None:
        assert CREDENTIALS_RULE in prompt_starting(PROMPT_STARTS[name])

    def test_the_account_prompt_never_signs_in_for_the_user(self) -> None:
        text = prompt_starting(PROMPT_STARTS["account"])

        assert "never try to sign in for me" in text
        assert "never ask for or handle a password, code, or token" in text

    def test_the_remove_prompt_keeps_the_users_data_and_leaves_deleting_it_to_them(self) -> None:
        text = prompt_starting(PROMPT_STARTS["remove"])

        assert "Do not delete the repository folder" in text
        assert "deleting them is my choice" in text


class TestTheInstallAndUpdatePromptsRunOnlyTheRightCheckout:
    @pytest.mark.parametrize("name", ["install", "update"])
    def test_an_existing_checkout_is_verified_in_order_before_anything_is_pulled_or_run(self, name: str) -> None:
        text = prompt_starting(PROMPT_STARTS[name])

        steps = [
            "remote get-url origin",
            "status --porcelain",
            "If either check fails, stop and tell me, and do not pull, sync, or run anything from that folder",
            "git pull --ff-only",
            "uv sync",
        ]
        positions = [text.index(step) for step in steps]
        assert positions == sorted(positions)
        assert REPOSITORY_ADDRESS in text
        assert "prints nothing" in text

    @pytest.mark.parametrize("name", ["install", "update"])
    def test_a_plain_pull_that_would_merge_is_never_used(self, name: str) -> None:
        assert not re.search(r"git pull(?! --ff-only)", prompt_starting(PROMPT_STARTS[name]))


class TestTheInstallAndUpdatePromptsDoNotTrustAnExitStatusAlone:
    @pytest.mark.parametrize("name", ["install", "update"])
    def test_doctor_is_run_even_when_install_exited_zero(self, name: str) -> None:
        text = prompt_starting(PROMPT_STARTS[name])

        assert "an older process can answer the health check while the new agent has failed" in text
        assert "usagedash doctor" in text
        assert "whatever install's exit status was" in text or "even if step 5 exited 0" in text

    def test_the_install_prompt_does_not_run_doctor_only_after_a_failure(self) -> None:
        assert "If step 5 exits nonzero" not in prompt_starting(PROMPT_STARTS["install"])

    def test_the_update_prompt_does_not_run_doctor_only_after_a_failure(self) -> None:
        assert "If it exits nonzero, run" not in prompt_starting(PROMPT_STARTS["update"])
