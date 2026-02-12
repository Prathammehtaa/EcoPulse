from google.cloud import storage

client = storage.Client()
print(client.list_buckets())
