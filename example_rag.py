from rag_pipeline import VideoRAG

def main():
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = VideoRAG()
    
    # Example queries relevant to Max Life Videos content
    queries = [
        "What are the main topics discussed in the videos?",
        "How to qualify leads in bank assurance?",
        "What is the NOPP criteria?",
        "How to build rapport with customers?",
        "What are the benefits of insurance?",
        "How to assess customer needs?",
        "What questions to ask customers?",
        "How to handle customer objections?",
        "What are the different types of insurance products?",
        "How to improve sales performance?"
    ]
    
    print(f"\nRunning {len(queries)} example queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        try:
            results = rag.query_videos(query, n_results=3)
            
            if results:
                print(f"\nFound {len(results)} relevant segments:")
                for j, result in enumerate(results, 1):
                    print(f"\n{j}. Video: {result['video_id']}")
                    print(f"   Timestamp: {result['timestamp']}")
                    print(f"   Text: {result['text']}")
                    print(f"   Similarity Score: {1 - result['similarity']:.4f}")
            else:
                print("No relevant segments found.")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 