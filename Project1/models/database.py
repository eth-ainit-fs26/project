from peewee import SqliteDatabase, Model, TextField
import json
import numpy as np

# Database configuration
database = SqliteDatabase('data/retail_project.db')

class BaseModel(Model):
    """Base model class that uses our SQLite database."""
    class Meta:
        database = database

class JSONField(TextField):
    """Custom field for storing JSON data (including numpy arrays)."""

    def db_value(self, value):
        """Convert Python value to database value."""
        if value is None:
            return value
        return json.dumps(value, cls=NpEncoder)

    def python_value(self, value):
        """Convert database value to Python value."""
        if value is None:
            return value
        return json.loads(value)

class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def initialize_database():
    """Initialize the database and create tables."""
    database = SqliteDatabase('data/retail_project.db')
    if database.is_closed():
        database.connect()

    from models.models import Customer, Item, Transaction, MarketingOperation, Delivery, ShopVisit

    database.create_tables([Customer, Item, Transaction, MarketingOperation, Delivery, ShopVisit], safe=True)
    return database

def close_database():
    """Close the database connection."""
    if not database.is_closed():
        database.close()

def reset_database():
    """Reset the database by dropping and recreating all tables."""
    database = SqliteDatabase('retail_project.db')
    if database.is_closed():
        database.connect()

    from models.models import Customer, Item, Transaction, MarketingOperation, Delivery, ShopVisit

    database.drop_tables([Customer, Item, Transaction, MarketingOperation, Delivery, ShopVisit], safe=True)
    database.create_tables([Customer, Item, Transaction, MarketingOperation, Delivery, ShopVisit], safe=True)