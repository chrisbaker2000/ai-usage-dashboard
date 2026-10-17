import os
from pathlib import Path

import pytest

from usagedash.errors import ConfigError
from usagedash.paths import directory_identity, same_directory


class TestDirectoryIdentity:
    def test_every_spelling_of_one_directory_is_stored_as_one_path(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (tmp_path / "link").symlink_to(home)

        identity = directory_identity(home)

        assert directory_identity(home / ".." / "home") == identity
        assert directory_identity(tmp_path / "link") == identity
        assert directory_identity(tmp_path / "later" / ".." / "home") == directory_identity(tmp_path / "home")
        assert ".." not in identity.parts

    def test_another_capitalization_is_stored_as_the_directory_entry(self, tmp_path: Path) -> None:
        home = tmp_path / "Home"
        home.mkdir()
        other = tmp_path / "HOME"
        if not same_directory(home, other):
            pytest.skip("this filesystem stores the two spellings as different directories")

        assert directory_identity(other) == directory_identity(home)
        assert directory_identity(other).name == "Home"

    def test_a_missing_tail_keeps_its_spelling_under_the_stored_prefix(self, tmp_path: Path) -> None:
        home = tmp_path / "Home"
        home.mkdir()
        other = tmp_path / "HOME"
        if not same_directory(home, other):
            pytest.skip("this filesystem stores the two spellings as different directories")

        identity = directory_identity(other / "no-such" / "child")

        assert identity.parts[-3:] == ("Home", "no-such", "child")

    def test_a_case_folded_name_that_is_not_the_same_directory_keeps_the_given_spelling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "Home"
        home.mkdir()
        real_samefile = os.path.samefile

        def different_letters_are_different(first: str, second: str) -> bool:
            if Path(first).name != Path(second).name:
                return False
            return real_samefile(first, second)

        monkeypatch.setattr(os.path, "samefile", different_letters_are_different)

        assert directory_identity(tmp_path / "HOME").name == "HOME"

    def test_a_path_that_cannot_be_resolved_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot identify"):
            directory_identity(tmp_path / "nul\x00here")


class TestSameDirectory:
    def test_two_spellings_of_one_existing_directory_are_one(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (tmp_path / "link").symlink_to(home)

        assert same_directory(home, tmp_path / "home" / ".." / "home")
        assert same_directory(home, tmp_path / "link")

    def test_two_existing_directories_are_not_one(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        assert not same_directory(tmp_path / "a", tmp_path / "b")

    @pytest.mark.parametrize("through", ["a missing directory", "a file"])
    def test_a_path_that_reaches_nothing_is_not_the_directory_that_exists(self, tmp_path: Path, through: str) -> None:
        home = tmp_path / "home"
        home.mkdir()
        detour = tmp_path / "detour"
        if through == "a file":
            detour.write_text("")

        assert not same_directory(detour / ".." / "home", home)

    def test_two_paths_neither_of_which_exists_are_compared_as_paths(self, tmp_path: Path) -> None:
        assert same_directory(tmp_path / "later" / ".." / "home", tmp_path / "home")
        assert not same_directory(tmp_path / "home", tmp_path / "other")

    def test_a_path_that_cannot_be_examined_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot tell whether"):
            same_directory(tmp_path / "nul\x00here", tmp_path)

    def test_a_path_that_cannot_be_compared_after_the_other_was_not_found_is_a_config_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError, match="cannot tell whether"):
            same_directory(tmp_path / "missing", tmp_path / "nul\x00here")
