"""Database connection handlers for Supabase and PostgreSQL."""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Supabase PostgreSQL client for cloud database operations."""
    
    def __init__(self):
        try:
            from supabase import create_client
            self.url = os.getenv("SUPABASE_URL")
            self.key = os.getenv("SUPABASE_KEY")
            
            if not self.url or not self.key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
            
            self.client = create_client(self.url, self.key)
        except ImportError:
            raise ImportError("supabase package not installed. Install with: pip install supabase")
    
    def insert_pitcher_data(self, df: pd.DataFrame, table_name: str = "pitcher_statcast"):
        """Insert pitcher data into Supabase."""
        try:
            data = df.to_dict('records')
            response = self.client.table(table_name).insert(data).execute()
            return response
        except Exception as e:
            print(f"Error inserting data: {str(e)}")
            return None
    
    def fetch_pitcher_data(self, table_name: str = "pitcher_statcast", limit: int = 1000):
        """Fetch pitcher data from Supabase."""
        try:
            response = self.client.table(table_name).select("*").limit(limit).execute()
            df = pd.DataFrame(response.data)
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

class PostgreSQLClient:
    """PostgreSQL client for local database operations."""
    
    def __init__(self):
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=os.getenv("PG_HOST", "localhost"),
                port=os.getenv("PG_PORT", 5432),
                database=os.getenv("PG_DATABASE", "mlb_biomechanics"),
                user=os.getenv("PG_USER", "postgres"),
                password=os.getenv("PG_PASSWORD", "")
            )
            self.cursor = self.conn.cursor()
        except ImportError:
            raise ImportError("psycopg2 package not installed. Install with: pip install psycopg2-binary")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {str(e)}")
    
    def insert_pitcher_data(self, df: pd.DataFrame, table_name: str = "pitcher_statcast"):
        """Insert pitcher data into PostgreSQL."""
        try:
            for _, row in df.iterrows():
                self.cursor.execute(f"""
                    INSERT INTO {table_name}
                    (pitcher_name, team, pitch_type, release_speed, release_spin_rate,
                     release_extension, pfx_z, pfx_x, release_pos_x, release_pos_z,
                     spin_efficiency, release_efficiency, arm_slot_proxy, movement_total,
                     velocity_diff, cluster_label, game_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(row))
            self.conn.commit()
            print(f"Inserted {len(df)} rows into {table_name}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting data: {str(e)}")
    
    def fetch_pitcher_data(self, table_name: str = "pitcher_statcast", limit: int = 1000):
        """Fetch pitcher data from PostgreSQL."""
        try:
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(self.cursor.fetchall(), columns=columns)
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.cursor.close()
            self.conn.close()
