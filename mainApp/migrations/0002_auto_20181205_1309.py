# Generated by Django 2.1.3 on 2018-12-05 10:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainApp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='graph',
            name='images_graph',
            field=models.ImageField(blank=True, db_column='Images_Graph', upload_to='profile_image'),
        ),
    ]
