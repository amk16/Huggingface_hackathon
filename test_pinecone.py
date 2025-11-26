#!/usr/bin/env python3
"""Quick test script to verify Pinecone connection"""
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Test connection
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    print("❌ PINECONE_API_KEY not found in environment")
    exit(1)

print("✅ PINECONE_API_KEY found")

pc = Pinecone(api_key=api_key)
index_name = os.getenv('PINECONE_INDEX_NAME', 'london-law-firms')

print(f"✅ Pinecone client initialized")
print(f"📋 Checking index: {index_name}")

if pc.has_index(index_name):
    print(f"✅ Index '{index_name}' exists!")
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"📊 Index stats: {stats}")
    print("\n🎉 Everything is set up correctly! You're ready to run the scraper.")
else:
    print(f"❌ Index '{index_name}' does not exist")
    print(f"\n   Please create it with:")
    print(f"   pc index create --name {index_name} --dimension 1536 --metric cosine --cloud aws --region us-east-1")

