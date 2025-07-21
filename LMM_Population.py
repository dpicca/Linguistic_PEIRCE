#!/usr/bin/env python3
"""
LMM_Population.py

This script converts Urban Dictionary entries into structured RDF triples
compatible with the LMM_L2.owl ontology. It extracts expressions (slang terms),
their senses (definitions), and objects (referents) from Urban Dictionary entries
and represents them using the Linguistic Meta-Model ontology.
"""

import csv
import zipfile
import re
import os
import urllib.parse
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD

# Define namespaces
LMM = Namespace("http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#")
EX = Namespace("http://example.org/urban#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# External knowledge bases
DBPEDIA = Namespace("http://dbpedia.org/resource/")
WIKIDATA = Namespace("http://www.wikidata.org/entity/")

def read_urban_dict_entries(zip_path):
    """
    Read Urban Dictionary entries from a zipped CSV file.
    
    Args:
        zip_path (str): Path to the zipped CSV file
        
    Returns:
        list: List of dictionaries containing Urban Dictionary entries
    """
    entries = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]  # Assuming there's only one file in the zip
        with zip_ref.open(csv_filename) as csv_file:
            # Decode bytes to string for CSV reader
            csv_text = csv_file.read().decode('utf-8')
            csv_reader = csv.DictReader(csv_text.splitlines())
            
            for row in csv_reader:
                entries.append({
                    'id': row['word_id'],
                    'expression': row['word'],
                    'upvotes': row['up_votes'],
                    'downvotes': row['down_votes'],
                    'author': row['author'],
                    'definition': row['definition']
                })
    
    return entries

def identify_object(definition):
    """
    Identify the object (referent) from the definition.
    This is a simplified implementation that looks for patterns like "a X" or "the X".
    
    Args:
        definition (str): The definition text
        
    Returns:
        tuple: (object_label, object_type, external_link)
    """
    # Common patterns in definitions
    patterns = [
        r"^(?:a|an|the)\s+([^\.;,]+)",  # Starts with "a", "an", or "the"
        r"(?:refers to|meaning)\s+(?:a|an|the)?\s*([^\.;,]+)",  # "refers to" or "meaning"
        r"([^\.;,]+)(?=\.|\s*$)"  # Last phrase before period or end of string
    ]
    
    for pattern in patterns:
        match = re.search(pattern, definition.lower())
        if match:
            obj = match.group(1).strip()
            
            # Try to identify common types of objects
            if re.search(r"person|individual|someone|somebody", obj):
                obj_type = "Person"
                ext_link = DBPEDIA.Person
            elif re.search(r"event|situation|occurrence|happening", obj):
                obj_type = "Event"
                ext_link = DBPEDIA.Event
            elif re.search(r"place|location|area|region", obj):
                obj_type = "Place"
                ext_link = DBPEDIA.Place
            elif re.search(r"action|activity|process", obj):
                obj_type = "Action"
                ext_link = DBPEDIA.Activity
            elif re.search(r"concept|idea|notion", obj):
                obj_type = "Concept"
                ext_link = DBPEDIA.Concept
            else:
                obj_type = "Thing"
                ext_link = None
                
            return (obj, obj_type, ext_link)
    
    # Default if no pattern matches
    return ("unspecified object", "Thing", None)

def sanitize_for_uri(text):
    """
    Sanitize text to be used in a URI.
    
    Args:
        text (str): Text to sanitize
        
    Returns:
        str: Sanitized text
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '_', text)
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Ensure the result is a valid URI component
    sanitized = urllib.parse.quote(sanitized)
    # Limit length to avoid excessively long URIs
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized

def entry_to_rdf(entry, graph):
    """
    Convert an Urban Dictionary entry to RDF triples.
    
    Args:
        entry (dict): Dictionary containing Urban Dictionary entry data
        graph (rdflib.Graph): RDF graph to add triples to
        
    Returns:
        None
    """
    # Create sanitized IDs for the entry components
    expr_text = sanitize_for_uri(entry['expression'].lower())
    expr_id = f"expr_{expr_text}"
    sense_id = f"sense_{expr_text}_1"
    
    # Identify the object from the definition
    obj_label, obj_type, ext_link = identify_object(entry['definition'])
    obj_text = sanitize_for_uri(obj_label)
    obj_id = f"obj_{obj_type}_{obj_text}"
    
    # Create the expression
    expr_uri = EX[expr_id]
    graph.add((expr_uri, RDF.type, LMM.LMM_Expression))
    graph.add((expr_uri, LMM.hasForm, Literal(entry['expression'])))
    
    # Add metadata
    # Handle non-numeric values in upvotes and downvotes
    try:
        upvotes = int(entry['upvotes'])
        graph.add((expr_uri, LMM.upvotes, Literal(upvotes, datatype=XSD.int)))
    except (ValueError, TypeError):
        # If conversion fails, store as string instead
        graph.add((expr_uri, LMM.upvotes, Literal(entry['upvotes'])))
        
    try:
        downvotes = int(entry['downvotes'])
        graph.add((expr_uri, LMM.downvotes, Literal(downvotes, datatype=XSD.int)))
    except (ValueError, TypeError):
        # If conversion fails, store as string instead
        graph.add((expr_uri, LMM.downvotes, Literal(entry['downvotes'])))
        
    graph.add((expr_uri, LMM.author, Literal(entry['author'])))
    
    # Create the sense
    sense_uri = EX[sense_id]
    graph.add((sense_uri, RDF.type, LMM.LMM_Sense))
    graph.add((sense_uri, LMM.hasDefinition, Literal(entry['definition'])))
    
    # Link expression to sense
    graph.add((expr_uri, LMM.hasSense, sense_uri))
    
    # Create the object
    obj_uri = EX[obj_id]
    graph.add((obj_uri, RDF.type, LMM.LMM_Object))
    graph.add((obj_uri, RDFS.label, Literal(obj_label)))
    
    # Link sense to object
    graph.add((sense_uri, LMM.hasObject, obj_uri))
    
    # Link to external resources if available
    if ext_link:
        graph.add((obj_uri, OWL.sameAs, ext_link))

def process_entries(entries, output_file, limit=100, batch_size=10000):
    """
    Process Urban Dictionary entries and write RDF triples to a file.
    Uses batch processing to optimize memory usage and performance.
    
    Args:
        entries (list): List of dictionaries containing Urban Dictionary entries
        output_file (str): Path to the output file
        limit (int, optional): Maximum number of entries to process. Defaults to 100.
        batch_size (int, optional): Number of entries to process in each batch. Defaults to 10000.
        
    Returns:
        None
    """
    # Initialize a counter for processed entries
    processed_count = 0
    
    # Calculate the number of batches
    num_entries = min(limit, len(entries))
    num_batches = (num_entries + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing up to {limit} entries in {num_batches} batches of {batch_size}...")
    
    # Process entries in batches
    for batch_num in range(num_batches):
        # Create a new RDF graph for this batch
        g = Graph()
        
        # Bind namespaces
        g.bind("lmm", LMM)
        g.bind("ex", EX)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("owl", OWL)
        g.bind("xsd", XSD)
        
        # Calculate start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_entries)
        
        print(f"Processing batch {batch_num + 1}/{num_batches} (entries {start_idx + 1}-{end_idx})...")
        
        # Process entries in this batch
        for i in range(start_idx, end_idx):
            entry = entries[i]
            try:
                entry_to_rdf(entry, g)
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries")
            except Exception as e:
                print(f"Error processing entry {entry['id']} ({entry['expression']}): {str(e)}")
        
        # Determine the output file for this batch
        if num_batches == 1:
            # If there's only one batch, write directly to the output file
            batch_output = output_file
            mode = "w"  # Write mode (overwrite existing file)
        else:
            # For multiple batches, use a temporary file for each batch
            batch_output = f"{output_file}.batch{batch_num + 1}"
            mode = "w"  # Write mode for each batch file
        
        # Serialize this batch to the appropriate file
        print(f"Serializing batch {batch_num + 1}...")
        # Use destination parameter to write directly to file
        g.serialize(destination=batch_output, format="turtle")
        
        print(f"Batch {batch_num + 1} written to {batch_output}")
    
    # If we processed multiple batches, combine them into a single output file
    if num_batches > 1:
        print(f"Combining {num_batches} batches into {output_file}...")
        with open(output_file, "w", encoding="utf-8") as outfile:
            # Write the header with namespace declarations (from the first batch)
            with open(f"{output_file}.batch1", "r", encoding="utf-8") as first_batch:
                # Copy the prefix declarations from the first batch
                for line in first_batch:
                    if line.startswith("@prefix"):
                        outfile.write(line)
                    elif not line.strip():
                        # Write an empty line after the prefixes
                        outfile.write(line)
                    else:
                        # Stop after the prefix declarations
                        break
            
            # Now append the triples from each batch file
            for batch_num in range(1, num_batches + 1):
                batch_file = f"{output_file}.batch{batch_num}"
                with open(batch_file, "r", encoding="utf-8") as batch:
                    # Skip the prefix declarations
                    for line in batch:
                        if not line.startswith("@prefix") and line.strip():
                            # Found the first non-prefix, non-empty line
                            outfile.write(line)
                            break
                    
                    # Copy the rest of the file
                    for line in batch:
                        outfile.write(line)
                
                # Remove the temporary batch file
                os.remove(batch_file)
        
        print(f"All batches combined into {output_file}")
    
    print(f"RDF triples written to {output_file}")

def main():
    """
    Main function to run the script.
    """
    # Input and output paths
    zip_path = "data/urbandict-word-defs.csv.zip"
    output_dir = "output"
    output_file = os.path.join(output_dir, "urban_dict_lmm.ttl")
    sample_file = os.path.join(output_dir, "sample_entry.ttl")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read entries from the zipped CSV file
    print(f"Reading Urban Dictionary entries from {zip_path}...")
    entries = read_urban_dict_entries(zip_path)
    print(f"Read {len(entries)} entries")
    
    # Process a sample entry for testing
    print("\nTesting with a sample entry...")
    sample_entry = entries[0]  # Use the first entry as a sample
    print(f"Sample entry: {sample_entry['expression']} - {sample_entry['definition']}")
    
    # Create a graph for the sample entry
    g = Graph()
    g.bind("lmm", LMM)
    g.bind("ex", EX)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    
    # Convert the sample entry to RDF
    entry_to_rdf(sample_entry, g)
    
    # Serialize the sample graph to Turtle format
    g.serialize(destination=sample_file, format="turtle")
    print(f"Sample RDF triples written to {sample_file}")
    
    # Print the sample RDF triples
    print("\nSample RDF triples:")
    print(g.serialize(format="turtle"))
    
    # Process a limited number of entries and write RDF triples
    print("\nConverting entries to RDF triples...")
    
    # Set the limit for the number of entries to process
    # This can be adjusted based on available memory and processing power
    entry_limit = 2580925
    
    # Process entries with the specified limit and batch size
    # Use a batch size of 10000 for efficient memory usage
    process_entries(entries, output_file, limit=entry_limit, batch_size=10000)
    
    print(f"\nProcessing complete. Processed {entry_limit} entries out of {len(entries)}.")
    print(f"RDF triples are available in {output_file}")
    print("\nExample usage of the generated RDF data:")
    print("1. Load the TTL file into a triple store or RDF database")
    print("2. Query the data using SPARQL")
    print("3. Link with other ontologies for enhanced semantic interoperability")

if __name__ == "__main__":
    main()