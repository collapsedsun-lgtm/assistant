-- Initial migration to create messages and list_items tables for the assistant

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    title TEXT,
    type TEXT NOT NULL,
    origin TEXT,
    destination TEXT,
    state TEXT,
    content TEXT,
    metadata TEXT,
    due_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS list_items (
    id INTEGER PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    position INTEGER,
    title TEXT,
    text TEXT,
    notes TEXT,
    sent_at DATETIME,
    checked INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
