from django.db import models

class FileStore(models.Model):
    fileid = models.BigAutoField(primary_key=True)
    filename = models.CharField(max_length=255)
    url = models.CharField(max_length=255)
    eventid = models.BigIntegerField()
    upload_time = models.DateTimeField(auto_now_add=True)
    isVectorized = models.BooleanField(default=False)

    class Meta:
        db_table = 'filestore'
