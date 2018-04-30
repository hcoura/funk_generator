# -*- coding: utf-8 -*-
import sqlite3


class SqlitePipeline(object):

    def __init__(self):
        self.conn = sqlite3.connect('songs.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        );
        """)

    def process_item(self, item, spider):
        self.cursor.execute('INSERT INTO songs (title, text) VALUES (?,?)',
                            (item['title'], item['text']))
        self.conn.commit()
        return item

    def __del__(self):
        self.conn.close()
