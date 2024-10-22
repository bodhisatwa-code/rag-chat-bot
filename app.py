# app.py

import os
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List
from lib import ProcessingConfig, SemanticSearchPipeline

console = Console()

def get_document_files(directory: str) -> List[str]:
    """Get all PDF and TXT files from the directory."""
    supported_extensions = {'.pdf', '.txt'}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise click.BadParameter(f"Directory {directory} does not exist")
    
    files = []
    for ext in supported_extensions:
        files.extend([str(f) for f in directory_path.glob(f"**/*{ext}")])
    
    return sorted(files)

def create_results_table(search_results: dict) -> Table:
    """Create a formatted table for search results."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Source", width=30)
    table.add_column("Content", width=50)

    for i, (doc, distance, metadata) in enumerate(zip(
        search_results['documents'][0],
        search_results['distances'][0],
        search_results['metadatas'][0]
    ), 1):
        score = f"{(1 - distance) * 100:.1f}%"
        content = (doc[:200] + "...") if len(doc) > 200 else doc
        source = Path(metadata['source']).name
        table.add_row(str(i), score, source, content)

    return table

@click.group()
def cli():
    """Semantic Search CLI for processing and searching documents."""
    pass

@cli.command()
@click.option('--docs-dir', default='documents', help='Directory containing documents to process')
@click.option('--chunk-size', default=500, help='Size of text chunks for processing')
@click.option('--chunk-overlap', default=20, help='Overlap between text chunks')
@click.option('--batch-size', default=100, help='Batch size for processing')
def process(docs_dir: str, chunk_size: int, chunk_overlap: int, batch_size: int):
    """Process documents from the specified directory."""
    try:
        # Initialize configuration
        config = ProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            processed_docs_file="processed_documents.json"
        )

        # Initialize pipeline
        pipeline = SemanticSearchPipeline(
            config=config,
            collection_name="documents",
            persist_directory="./chroma_persist"
        )

        # Get all documents from the directory
        files = get_document_files(docs_dir)
        
        if not files:
            console.print(f"[yellow]No supported documents found in {docs_dir}[/yellow]")
            return

        console.print(f"[green]Found {len(files)} documents to process[/green]")
        
        # Process documents
        results = pipeline.process_multiple_documents(files)
        
        # Display results
        success_count = sum(1 for v in results.values() if v)
        console.print(f"\n[bold green]Processing completed:[/bold green]")
        console.print(f"Successfully processed: {success_count}/{len(files)} documents")
        
        if success_count != len(files):
            failed = [path for path, success in results.items() if not success]
            console.print("\n[bold red]Failed documents:[/bold red]")
            for path in failed:
                console.print(f"- {path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()

@cli.command()
@click.argument('query')
@click.option('--num-results', '-n', default=5, help='Number of results to return')
@click.option('--min-score', '-s', default=0.0, help='Minimum similarity score (0-1)')
def search(query: str, num_results: int, min_score: float):
    """Search through processed documents."""
    try:
        # Initialize pipeline with default config
        config = ProcessingConfig()
        pipeline = SemanticSearchPipeline(
            config=config,
            collection_name="documents",
            persist_directory="./chroma_persist"
        )
        
        # Perform search
        results = pipeline.search_documents(query, n_results=num_results)
        
        # Filter results by score if needed
        if min_score > 0:
            filtered_results = {
                'documents': [[]], 'distances': [[]], 'metadatas': [[]]
            }
            for i, distance in enumerate(results['distances'][0]):
                if (1 - distance) >= min_score:
                    filtered_results['documents'][0].append(results['documents'][0][i])
                    filtered_results['distances'][0].append(distance)
                    filtered_results['metadatas'][0].append(results['metadatas'][0][i])
            results = filtered_results

        if not results['documents'][0]:
            console.print("[yellow]No results found[/yellow]")
            return

        # Display results
        console.print(f"\n[bold green]Search results for:[/bold green] {query}")
        table = create_results_table(results)
        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()

if __name__ == '__main__':
    cli()