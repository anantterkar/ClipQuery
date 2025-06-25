from rag_pipeline import VideoRAG

def main():
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = VideoRAG()
    
    print("\n" + "="*60)
    print("Interactive Video Query System")
    print("="*60)
    print("Type your queries to search through the video transcripts.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'help' for example queries.")
    print("="*60)
    
    example_queries = [
        "How to qualify leads in bank assurance?",
        "What is the NOPP criteria?",
        "How to build rapport with customers?",
        "What are the benefits of insurance?",
        "How to assess customer needs?",
        "What questions to ask customers?",
        "How to handle customer objections?",
        "What are the different types of insurance products?",
        "How to improve sales performance?",
        "What are the main topics discussed in the videos?"
    ]
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\nExample queries you can try:")
                for i, example in enumerate(example_queries, 1):
                    print(f"{i}. {example}")
                continue
                
            if not query:
                print("Please enter a valid query.")
                continue
            
            print(f"\nSearching for: '{query}'")
            print("-" * 40)
            
            results = rag.query_videos(query, n_results=5)
            
            if results:
                print(f"\nFound {len(results)} relevant segments:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Video: {result['video_id']}")
                    print(f"   Timestamp: {result['timestamp']}")
                    print(f"   Text: {result['text']}")
                    print(f"   Similarity Score: {1 - result['similarity']:.4f}")
            else:
                print("No relevant segments found. Try a different query.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 