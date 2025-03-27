from .models import FileStore
from .vectorize import get_face_embeddings
from milvus_integration.tasks import addVectorsToMilvus

async def vectorizeImage(message_list):
    for item in message_list:
        fileUrl =item['url']
        try:
            vecotrs = await get_face_embeddings(fileUrl)
            print(f"found {len(vecotrs)} faces in {fileUrl}")
            for vector in vecotrs:
                addVectorsToMilvus(item['fileid'],vector,item['eventid'])
            await FileStore.objects.filter(fileid=item['fileid']).aupdate(isVectorized=1)
        except Exception as e: 
            print(f"Error processing image: {e}")
    print("Processing image...")