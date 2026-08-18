import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


class _FakePyMySQLSaver:
    opened_uris = []
    instance = None

    @classmethod
    @contextmanager
    def from_conn_string(cls, uri):
        cls.opened_uris.append(uri)
        cls.instance = MagicMock()
        yield cls.instance


class MySQLCheckpointingTest(unittest.TestCase):
    def setUp(self):
        _FakePyMySQLSaver.opened_uris = []
        _FakePyMySQLSaver.instance = None

    def test_requires_an_explicit_mysql_uri(self):
        from src.infrastructure.checkpointing.mysql import (
            open_mysql_checkpointer,
        )

        with self.assertRaisesRegex(ValueError, "MYSQL_CHECKPOINT_URI"):
            with open_mysql_checkpointer("   "):
                self.fail("empty configuration must fail before opening MySQL")

    def test_open_is_lazy_and_setup_is_opt_in(self):
        from src.infrastructure.checkpointing import mysql as mysql_module

        with patch.object(
            mysql_module,
            "_load_pymysql_saver",
            return_value=_FakePyMySQLSaver,
        ) as loader:
            manager = mysql_module.open_mysql_checkpointer(
                "mysql://app:secret@localhost:3306/rag_checkpoint"
            )

            loader.assert_not_called()
            self.assertEqual(_FakePyMySQLSaver.opened_uris, [])
            with manager as checkpointer:
                loader.assert_called_once_with()
                checkpointer.configure_database_collation.assert_called_once_with()
                checkpointer.setup.assert_not_called()

        self.assertEqual(
            _FakePyMySQLSaver.opened_uris,
            ["mysql://app:secret@localhost:3306/rag_checkpoint"],
        )

    def test_setup_runs_only_when_explicitly_requested(self):
        from src.infrastructure.checkpointing import mysql as mysql_module

        with patch.object(
            mysql_module,
            "_load_pymysql_saver",
            return_value=_FakePyMySQLSaver,
        ):
            with mysql_module.open_mysql_checkpointer(
                "mysql://app:secret@localhost:3306/rag_checkpoint",
                setup=True,
            ) as checkpointer:
                checkpointer.setup.assert_called_once_with()

    def test_checkpoint_select_uses_the_database_collation_for_json_table(self):
        from src.infrastructure.checkpointing.mysql import (
            _apply_checkpoint_database_collation,
        )

        sql = (
            "SELECT * FROM JSON_TABLE(data, '$[*]' COLUMNS ("
            "channel VARCHAR(150) CHARACTER SET utf8mb4 PATH '$')) AS channels"
        )

        result = _apply_checkpoint_database_collation(
            sql,
            "utf8mb4_unicode_ci",
        )

        self.assertIn(
            "channel VARCHAR(150) CHARACTER SET utf8mb4 "
            "COLLATE utf8mb4_unicode_ci PATH '$'",
            result,
        )

    def test_checkpoint_collation_rejects_an_unsafe_identifier(self):
        from src.infrastructure.checkpointing.mysql import (
            _apply_checkpoint_database_collation,
        )

        with self.assertRaisesRegex(ValueError, "collation"):
            _apply_checkpoint_database_collation(
                "channel VARCHAR(150) CHARACTER SET utf8mb4 PATH '$'",
                "utf8mb4_unicode_ci; DROP TABLE checkpoints",
            )

    def test_close_session_targets_only_the_explicit_thread(self):
        from src.infrastructure.checkpointing.mysql import close_session

        checkpointer = MagicMock()
        close_session(checkpointer, "thread-42")

        checkpointer.delete_thread.assert_called_once_with("thread-42")

    def test_close_session_rejects_an_empty_thread_id(self):
        from src.infrastructure.checkpointing.mysql import close_session

        checkpointer = MagicMock()
        with self.assertRaisesRegex(ValueError, "thread_id"):
            close_session(checkpointer, "  ")

        checkpointer.delete_thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
