import psycopg2

from datetime import datetime, timezone

class DBManager:
    def __init__(self, config):
        self.enabled = config.get("db_enabled", "false").lower() == "true"
        if not self.enabled:
            return

        self.conn = psycopg2.connect(
            host=config["db_host"],
            port=config["db_port"],
            database=config["db_name"],
            user=config["db_user"],
            password=config["db_password"]
        )
        self.conn.autocommit = True
        self.schema = config.get("db_schema", "public")
        


    def day_epoch(self):
            """
            Returns epoch seconds for today's date at 00:00:00 (UTC)
            """
            now = datetime.now(timezone.utc)
            day_start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
            return int(day_start.timestamp() * 1000)
    
    def upsert_productivity(
        self,
        camera_id,
        worker_name,
        active_sec,
        idle_sec
    ):
        if not self.enabled:
            return

        total = active_sec + idle_sec
        inc_time = self.day_epoch()

        query = f"""
        INSERT INTO {self.schema}.worker_productivity
        (camera_id, worker_name, inc_time, active_seconds, idle_seconds, total_seconds)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (camera_id, worker_name, inc_time)
        DO UPDATE SET
            active_seconds = EXCLUDED.active_seconds,
            idle_seconds = EXCLUDED.idle_seconds,
            total_seconds = EXCLUDED.total_seconds,
            last_updated = CURRENT_TIMESTAMP;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (
                camera_id,
                worker_name,
                inc_time,
                active_sec,
                idle_sec,
                total
            ))


    def delete_worker(self, camera_id, worker_name):
        if not self.enabled:
            return

        inc_time = self.day_epoch()

        query = f"""
        DELETE FROM {self.schema}.worker_productivity
        WHERE camera_id = %s
        AND worker_name = %s
        AND inc_time = %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (camera_id, worker_name, inc_time))
