import sys
from pathlib import Path
from src.search_engine import SearchEngine


def print_divider():
    print("\n" + "=" * 60 + "\n")


def main():
    print_divider()
    print(" 🔍 LOCAL NOTES SEMANTIC SEARCH ENGINE")
    print_divider()

    # Initialize the engine
    engine = SearchEngine()

    # Initial indexing build/load
    engine.build_or_load_index()

    print("\nCommands:")
    print("  type your query to search your notes")
    print("  :reindex  - force reload and re-encode notes from disk")
    print("  :exit     - close the application")
    print_divider()

    while True:
        try:
            # Interactive prompt
            query = input("search_notes ❯ ").strip()

            if not query:
                continue

            if query.lower() == ":exit":
                print("Goodbye!")
                break

            if query.lower() == ":reindex":
                print_divider()
                print("🔄 Re-scanning files and updating embeddings...")
                engine.build_or_load_index(force_reindex=True)
                print_divider()
                continue

            # Execute semantic lookup
            print("\nSearching...")
            results = engine.search(query, top_k=3)

            print_divider()
            print(f"Top matches for: '{query}'")
            print_divider()

            if not results:
                print("No matching notes found.")
            else:
                for rank, res in enumerate(results, 1):
                    # Style formatting for clear reading
                    print(f"[{rank}] Match Score: {res['score']:.4f}")
                    print(f"    Source: {res['source']}")
                    print(f"    Content:\n    {'-'*20}")
                    # Indent content slightly for visual separation
                    indented_content = "\n".join(
                        f"    {line}" for line in res["content"].splitlines()
                    )
                    print(indented_content)
                    print(f"    {'-'*20}\n")

            print_divider()

        except KeyboardInterrupt:
            # Handle Ctrl+C cleanly
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")


if __name__ == "__main__":
    main()