# LMM Population Script

## Overview

LMM_Population.py is a Python script that transforms Urban Dictionary entries into RDF triples compatible with the Linguistic Meta-Model (LMM) L2 ontology.
The script parses Urban Dictionary data (in zipped CSV format), extracts relevant linguistic information (expression, sense, object/referent), and generates RDF/Turtle output suitable for use in Semantic Web applications, triple stores, or ontology-based research.

---

## Features

- Automatic extraction of expressions (slang terms), senses (definitions), and objects (referents) from Urban Dictionary entries.
- Heuristic object identification from natural language definitions.
- Mapping to LMM ontology with support for expression, sense, and object classes.
- Batch processing for large datasets.
- External linking to DBpedia and Wikidata entities (when possible).
- Output in Turtle (TTL) format for easy integration into Semantic Web tools.

---

## File Structure

- LMM_Population.py – Main Python script.
- data/urbandict-word-defs.csv.zip – Input: zipped CSV file with Urban Dictionary entries.
- output/urban_dict_lmm.ttl – Main RDF output (Turtle).
- output/sample_entry.ttl – Example RDF output for a single entry.

---

## Requirements

- Python 3.x
- rdflib (for RDF handling)

Install dependencies:

```bash
pip install rdflib
```

---

## Usage

1. **Prepare your input data**  
   Place the zipped CSV file (e.g., urbandict-word-defs.csv.zip) in the data/ directory. The CSV file should have columns:  
   - word_id
   - word
   - up_votes
   - down_votes
   - author
   - definition

2. **Run the script**
   ```bash
   python LMM_Population.py
   ```

3. **Outputs**
   - A sample Turtle file for the first entry at: output/sample_entry.ttl
   - The main RDF/Turtle file (all entries, possibly batched) at: output/urban_dict_lmm.ttl

---

## Functionality Details

- read_urban_dict_entries: Reads Urban Dictionary data from zipped CSV.
- identify_object: Uses simple NLP heuristics and regex to extract the object (referent) from the definition and link to external resources when possible.
- entry_to_rdf: Maps the entry (expression, sense, object) to RDF triples following the LMM ontology.
- process_entries: Handles batching, file writing, and merging of output Turtle files for large datasets.
- main: Orchestrates the process, prints sample output, and guides the user for further use.

---

## Example Output (Turtle)

```turtle
@prefix lmm: <http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#> .
@prefix ex: <http://example.org/urban#> .

ex:expr_twomp a lmm:LMM_Expression ;
    lmm:hasForm "twomp" ;
    lmm:hasSense ex:sense_twomp_1 ;
    lmm:upvotes 14 ;
    lmm:downvotes 57 ;
    lmm:author "84dc1383" .

ex:sense_twomp_1 a lmm:LMM_Sense ;
    lmm:hasDefinition "a twenty dollar bill. Jackson's on it." ;
    lmm:hasObject ex:obj_Thing_twenty_dollar_bill .

ex:obj_Thing_twenty_dollar_bill a lmm:LMM_Object ;
    rdfs:label "twenty dollar bill" ;
    owl:sameAs <http://dbpedia.org/resource/United_States_twenty-dollar_bill> .
```

---

## Customization

- Change the batch size or entry limit:  
  Modify the entry_limit and batch_size parameters in the main() function to suit your memory and performance requirements.
- Heuristic improvement:  
  You can enhance the identify_object function for better semantic extraction or link to other ontologies.
- Different CSV structure:  
  If your CSV uses different field names, edit the read_urban_dict_entries function accordingly.

---

## License

This script is provided as-is for academic and research use.  
Feel free to adapt and extend for your ontology population needs.

---
