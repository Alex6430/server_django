from django.db import models
from django.db import connection
import pyodbc


class table_Log(models.Model):
    id_log = models.AutoField(db_column='ID_Log', primary_key=True)  # Field name made lowercase.
    Date_log = models.DateTimeField(db_column='Date')
    Valeu_log = models.FloatField(db_column='Value')

    class Meta:
        managed = True
        db_table = 'Log'


class Graph(models.Model):
    id_graph = models.AutoField(db_column='ID_Graph', primary_key=True)  # Field name made lowercase.
    images_graph = models.ImageField(db_column='Images_Graph', upload_to='profile_image',
                                     blank=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Graph'


class Matrixsize(models.Model):
    id_km = models.AutoField(db_column='ID_KM', primary_key=True)  # Field name made lowercase.
    id_k = models.IntegerField(db_column='ID_K')  # Field name made lowercase.
    id_m = models.IntegerField(db_column='ID_M')  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'MatrixSize'


class Nu(models.Model):
    id_nu = models.AutoField(db_column='ID_Nu', primary_key=True)  # Field name made lowercase.
    value_nu = models.FloatField(db_column='Value_Nu')  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Nu'


class Q(models.Model):
    id_q = models.AutoField(db_column='ID_Q', primary_key=True)  # Field name made lowercase.
    value_q = models.FloatField(db_column='Value_Q')  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'Q'


class R(models.Model):
    id_r = models.AutoField(db_column='ID_R', primary_key=True)  # Field name made lowercase.
    value_r = models.FloatField(db_column='Value_R')  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'R'


def search():
    # create a cursor
    cur = connection.cursor()
    cur.execute("EXEC proc_res")
    # execute the stored procedure passing in
    # search_string as a parameter
    # cur.callproc('proc_for_project', [name_str])
    # grab the results
    results = cur.fetchone()
    cur.close()
    # wrap the results up into Document domain objects
    return results


def backup():
    # create a cursor
    connection1 = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                                 server='DESKTOP-UP524EV', database='master',
                                 trusted_connection='yes', autocommit=True)
    backup = '''
    BACKUP DATABASE [new1 диплом] 
	TO  DISK = N'C:\Program Files\Microsoft SQL Server\MSSQL14.MSSQLSERVER\MSSQL\Backup\рез_коп_бд.bak' WITH NOFORMAT, 
	NOINIT,  
	NAME = N'new1 диплом-Полная База данных Резервное копирование', 
	SKIP, 
	NOREWIND, 
	NOUNLOAD,  
	STATS = 10
	'''
    cursor = connection1.cursor().execute(backup)
    connection1.close()


def restore():
    # create a cursor
    connection2 = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                                 server='DESKTOP-UP524EV', database='master',
                                 trusted_connection='yes', autocommit=True)

    connection2.cursor().execute("EXEC restor")
    connection2.commit()
    connection2.close()
    print('close')
