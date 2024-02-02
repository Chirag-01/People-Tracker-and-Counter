import cv2
from ultralytics import YOLO
import os
import threading
import time
# import boto3
import psycopg2 as spg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
import json
import torch
from final_react import object_tracking
 
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASS = "root"

def connect_to_db(DB_NAME="new db"):
    # Connect to the DATABASE
    con = spg.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port='5432'
    )
    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Cursor
    cur = con.cursor()
    cur.execute("select version()")
    data = cur.fetchone()
    print("Connection established to:", data)
    print(con)
    print(cur)
    return con, cur
 
def create_table_if_not_exists(cur):

    create_table_sql = """

    CREATE TABLE IF NOT EXISTS real_time_person_count (

        id SERIAL PRIMARY KEY,

        class1 TEXT,

        production_house TEXT,

        camera_unit TEXT,

        date1 date,

        time1 time,

        area TEXT,

        person_count INT
    );

    """

if __name__ == '__main__':

    # devices = json.loads(open('./camera.json').read())
    devices = json.loads(open('./camera.json').read())

    threads = []

 
    for plant, plantData in devices.items():

        for productionHouse, productionHouseData in plantData.items():

            for areaType, areaData in productionHouseData.items():

                stream_link = areaData['streamLink']


                after_at = stream_link.split('@')[1]
#
                ip_address = after_at.split(':')[0]
 
                ppe_list = areaData.get("ppeList", [])
                print(areaData)

 
 
                thread = threading.Thread(target=object_tracking,

                                          args=(areaData,productionHouse,ip_address,areaType))


                threads.append(thread)
 
    for thread in threads:

        thread.start()
 
    for thread in threads:

        thread.join()