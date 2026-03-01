# debug_db.py
import chromadb
client = chromadb.PersistentClient(path="./my_multimodal_db")
col = client.get_collection(name="multimodal_knowledge")

print(f"📊 数据库当前总条目数: {col.count()}")
# 查看最后 5 条数据
peek = col.peek(limit=5)
for i in range(len(peek['ids'])):
    print(f"ID: {peek['ids'][i]} | Type: {peek['metadatas'][i].get('type')} | Source: {peek['metadatas'][i].get('source')}")