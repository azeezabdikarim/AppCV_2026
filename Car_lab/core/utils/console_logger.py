#!/usr/bin/env python3

import time
from datetime import datetime
from collections import deque
import threading

class ConsoleLogger:
    """
    Simple console logger for robot operation messages
    Stores messages in memory for web console display
    """
    
    def __init__(self, max_messages=100):
        """Initialize logger with message history limit"""
        self.messages = deque(maxlen=max_messages)
        self.lock = threading.Lock()
        
    def _add_message(self, level, message):
        """Add a timestamped message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        with self.lock:
            self.messages.append(log_entry)
        
        # Also print to terminal for debugging
        print(f"[{timestamp}] {message}")
    
    def info(self, message):
        """Log an info message"""
        self._add_message('INFO', message)
    
    def warning(self, message):
        """Log a warning message"""
        self._add_message('WARN', message)
    
    def error(self, message):
        """Log an error message"""
        self._add_message('ERROR', message)
    
    def stop(self, message):
        """Log a stop event (special category)"""
        self._add_message('STOP', message)
    
    def get_recent_messages(self, limit=50):
        """Get recent messages for web console"""
        with self.lock:
            # Return most recent messages (up to limit)
            recent = list(self.messages)[-limit:]
            return recent
    
    def clear(self):
        """Clear all messages"""
        with self.lock:
            self.messages.clear()

# Global console logger instance
console_logger = ConsoleLogger()