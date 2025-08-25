"""
Database connection management for PostgreSQL on Railway.
Handles connection pooling, session management, and database initialization.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from urllib.parse import urlparse
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and sessions."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
    def initialize(self, database_url: str = None):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection string. If None, reads from environment.
        """
        try:
            # Get database URL from environment if not provided
            if database_url is None:
                database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_PRIVATE_URL')
                
            if not database_url:
                raise ValueError("No database URL provided. Set DATABASE_URL environment variable.")
                
            # Parse and validate URL
            parsed_url = urlparse(database_url)
            if parsed_url.scheme not in ['postgres', 'postgresql']:
                # Railway sometimes uses postgres:// which needs to be postgresql://
                if parsed_url.scheme == 'postgres':
                    database_url = database_url.replace('postgres://', 'postgresql://', 1)
                else:
                    raise ValueError(f"Invalid database scheme: {parsed_url.scheme}")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False,  # Set to True for SQL debugging
                future=True
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                future=True
            )
            
            # Test connection
            self._test_connection()
            
            self._initialized = True
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
            
    def _test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
            
    def create_tables(self, drop_first: bool = False):
        """
        Create database tables.
        
        Args:
            drop_first: If True, drops existing tables first (USE WITH CAUTION)
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        try:
            if drop_first:
                logger.warning("Dropping all existing tables")
                Base.metadata.drop_all(bind=self.engine)
                
            logger.info("Creating database tables")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
            
    def get_session(self):
        """Get a new database session."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
        
    @contextmanager
    def session_scope(self):
        """
        Context manager for database sessions with automatic commit/rollback.
        
        Usage:
            with db.session_scope() as session:
                session.add(obj)
                # Automatic commit on success, rollback on exception
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    def health_check(self):
        """
        Perform health check on database connection.
        
        Returns:
            dict: Health check results
        """
        try:
            with self.engine.connect() as connection:
                # Test basic connectivity
                result = connection.execute(text("SELECT version()"))
                db_version = result.fetchone()[0]
                
                # Test table existence
                result = connection.execute(text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                ))
                table_count = result.fetchone()[0]
                
                # Connection pool status
                pool = self.engine.pool
                pool_status = {
                    'size': pool.size(),
                    'checked_in': pool.checkedin(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow(),
                    'invalid': pool.invalid()
                }
                
                return {
                    'status': 'healthy',
                    'database_version': db_version,
                    'table_count': table_count,
                    'connection_pool': pool_status,
                    'initialized': self._initialized
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'initialized': self._initialized
            }
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Global database manager instance
db = DatabaseManager()

def init_database(database_url: str = None, create_tables: bool = True):
    """
    Initialize the global database manager.
    
    Args:
        database_url: PostgreSQL connection string
        create_tables: Whether to create tables if they don't exist
    """
    try:
        db.initialize(database_url)
        
        if create_tables:
            db.create_tables(drop_first=False)
            
        # Initialize launch sites if they don't exist
        _initialize_launch_sites()
        
        logger.info("Database initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def _initialize_launch_sites():
    """Initialize launch sites data if not present."""
    from .models import LaunchSite
    
    launch_sites_data = [
        {
            'site_code': 'KSC',
            'site_name': 'Kennedy Space Center',
            'latitude': 28.5721,
            'longitude': -80.648,
            'altitude_m': 3,
            'timezone': 'America/New_York'
        },
        {
            'site_code': 'VSFB',
            'site_name': 'Vandenberg Space Force Base',
            'latitude': 34.742,
            'longitude': -120.5724,
            'altitude_m': 367,
            'timezone': 'America/Los_Angeles'
        },
        {
            'site_code': 'CCAFS',
            'site_name': 'Cape Canaveral Space Force Station',
            'latitude': 28.3922,
            'longitude': -80.6077,
            'altitude_m': 16,
            'timezone': 'America/New_York'
        }
    ]
    
    try:
        with db.session_scope() as session:
            for site_data in launch_sites_data:
                existing_site = session.query(LaunchSite).filter_by(
                    site_code=site_data['site_code']
                ).first()
                
                if not existing_site:
                    site = LaunchSite(**site_data)
                    session.add(site)
                    logger.info(f"Added launch site: {site_data['site_code']}")
                    
    except Exception as e:
        logger.error(f"Failed to initialize launch sites: {e}")

def get_database_url_from_railway():
    """
    Helper function to construct database URL from Railway environment variables.
    Railway provides these variables automatically.
    """
    # Railway provides these environment variables
    db_host = os.getenv('PGHOST')
    db_port = os.getenv('PGPORT', '5432')
    db_name = os.getenv('PGDATABASE')
    db_user = os.getenv('PGUSER')
    db_password = os.getenv('PGPASSWORD')
    
    if all([db_host, db_name, db_user, db_password]):
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Fallback to DATABASE_URL or DATABASE_PRIVATE_URL
    return os.getenv('DATABASE_URL') or os.getenv('DATABASE_PRIVATE_URL')