# Code by AkinoAlice@TyrantRey

import sqlite3

from pathlib import Path


class Database:
    def __init__(self, db_path: str | Path = "posture.sqlite3"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = self._dict_factory
        self._create_tables()

    def _dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def _create_tables(self):
        cursor = self.conn.cursor()
        # posture table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS posture (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            posture_name TEXT NOT NULL,
            video_path TEXT,
            npy_path TEXT
        );
        """)
        # score table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS score (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time INTEGER,
            score INTEGER,
            video_path TEXT
        );
        """)
        self.conn.commit()

    def insert_posture(self, posture_name: str, video_path: str, npy_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO posture (posture_name, video_path, npy_path) VALUES (?, ?, ?)",
            (posture_name, video_path, npy_path),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_score(self, time: int, score: int, video_path: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO score (time, score, video_path) VALUES (?, ?, ?)",
            (time, score, video_path),
        )
        self.conn.commit()
        return cursor.lastrowid

    def fetch_all_postures(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM posture")
        return cursor.fetchall()

    def fetch_scores_by_video(self, video_path: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM score WHERE video_path = ?", (video_path,))
        return cursor.fetchall()

    def close(self):
        self.conn.close()


sqlite3_database = Database()

if __name__ == "__main__":
    db = Database()

    # Example usage
    db.insert_posture("squat", "videos/squat.mp4", "npy/squat.npy")
    db.insert_score(12, 95, "videos/squat.mp4")

    print(db.fetch_all_postures())
    print(db.fetch_scores_by_video("videos/squat.mp4"))

    db.close()
