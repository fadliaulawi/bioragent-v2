"""Planner-facing biomedical tools and dispatch (`format_tool_output`, `run_tool`)."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union
from urllib.parse import quote

from helpers import (
    NCBI_DATASETS_GENE_SYMBOL_BASE,
    bioontology_api_key,
    choose_best_gene_like_hit,
    extract_dna_sequence,
    extract_hit_description,
    fetch_blast_json,
    fetch_data,
    fetch_gene_aliases,
    find_hits_recursive,
    get_phenotype_id,
    guess_gene_symbol_from_title,
    ncbi_gene_dataset_headers,
    poll_blast_ready,
    submit_blast,
)

# —— Display names must match prompts.TOOL_CATALOG ——

PHENOTYPES_INFO_EXTRACTOR = "Phenotypes Info Extractor"
PHENOTYPES_PARENTS_EXTRACTOR = "Phenotypes Parents Extractor"
PHENOTYPES_CHILDRENS_EXTRACTOR = "Phenotypes Childrens Extractor"
PHENOTYPES_DISEASE_EXTRACTOR = "Phenotypes Disease Extractor"
PHENOTYPES_GENE_EXTRACTOR = "Phenotypes Gene Extractor"
GENE_PHENOTYPES_EXTRACTOR = "Gene Phenotypes Extractor"
GENE_DISEASES_EXTRACTOR = "Gene Diseases Extractor"
DISEASE_PHENOTYPES_EXTRACTOR = "Disease Phenotypes Extractor"
PROTEIN_INFORMATION_EXTRACTOR = "Protein Information Extractor"
GENE_INFORMATION_TOOL = "Gene Information Tool"
DISEASE_INFORMATION_EXTRACTOR = "Disease Information Extractor"
DISEASE_GENE_EXTRACTOR = "Disease Gene Extractor"
SNP_INFORMATION_EXTRACTOR = "SNP Information Extractor"
SEQUENCE_INFORMATION_EXTRACTOR = "Sequence Information Extractor"


def run_phenotypes_info(phenotype_term: str) -> str:
    phenotype_id = get_phenotype_id(phenotype_term)
    url_id = f"https://ontology.jax.org/api/hp/terms/{quote(str(phenotype_id))}"
    data = fetch_data(url_id)
    return data if data else "Not Found"


def run_phenotypes_parents(phenotype_term: str) -> str:
    phenotype_id = get_phenotype_id(phenotype_term)
    url_id = f"https://ontology.jax.org/api/hp/terms/{quote(str(phenotype_id))}/parents"
    data = fetch_data(url_id)
    if data:
        return [
            {
                "id": item["id"],
                "name": item["name"],
                "descendantCount": item["descendantCount"],
            }
            for item in data
        ]
    return "Not Found"


def run_phenotypes_children(phenotype_term: str) -> str:
    phenotype_id = get_phenotype_id(phenotype_term)
    url_id = f"https://ontology.jax.org/api/hp/terms/{quote(str(phenotype_id))}/children"
    data = fetch_data(url_id)
    if data:
        return [
            {
                "id": item["id"],
                "name": item["name"],
                "descendantCount": item["descendantCount"],
            }
            for item in data
        ]
    return "Not Found"


def run_phenotypes_disease(phenotype_term: str) -> Union[List[str], str]:
    phenotype_id = get_phenotype_id(phenotype_term)
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(str(phenotype_id))}"
    data = fetch_data(url_annotation)
    if data and "diseases" in data:
        return data["diseases"]
    return "Not Found"


def run_phenotypes_gene(phenotype_term: str) -> Union[List[str], str]:
    phenotype_id = get_phenotype_id(phenotype_term)
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(str(phenotype_id))}"
    data = fetch_data(url_annotation)
    if data and "genes" in data:
        return data["genes"]
    return "Not Found"


def run_gene_phenotypes(gene_term: str) -> Union[List[str], str]:
    gene_id = None

    def is_gene_id(s: str) -> bool:
        return s.startswith("NCBIGene:") and s[9:].isdigit()

    if is_gene_id(gene_term):
        gene_id = gene_term
    elif gene_term.isdigit():
        gene_id = f"NCBIGene:{gene_term}"
    else:
        url_name = f"https://ontology.jax.org/api/network/search/gene?q={quote(gene_term)}&page=0&limit=10"
        response = fetch_data(url_name)
        if response and "results" in response and len(response["results"]) > 0:
            gene_id = response["results"][0]["id"]
        else:
            return "Not Found"
    if gene_id is None:
        return "Not Found"
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(str(gene_id))}"
    response = fetch_data(url_annotation)
    if response and "phenotypes" in response:
        return response["phenotypes"]
    return "Not Found"


def run_gene_diseases(gene_term: str) -> Union[List[str], str]:
    gene_id = None

    def is_gene_id(s: str) -> bool:
        return s.startswith("NCBIGene:") and s[9:].isdigit()

    if is_gene_id(gene_term):
        gene_id = gene_term
    elif gene_term.isdigit():
        gene_id = f"NCBIGene:{gene_term}"
    else:
        url_name = f"https://ontology.jax.org/api/network/search/gene?q={quote(gene_term)}&page=0&limit=10"
        response = fetch_data(url_name)
        if response and "results" in response and len(response["results"]) > 0:
            gene_id = response["results"][0]["id"]
        else:
            return "Not Found"
    if gene_id is None:
        return "Not Found"
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(str(gene_id))}"
    response = fetch_data(url_annotation)
    if response and "diseases" in response:
        return response["diseases"]
    return "Not Found"


def run_disease_phenotypes(disease_term: str) -> Union[str, None]:
    disease_id = None

    def is_disease_id(s: str) -> bool:
        return (s.startswith("OMIM:") and s[5:].isdigit()) or (
            s.startswith("ORPHA:") and s[6:].isdigit()
        )

    if not disease_term.isdigit():
        if is_disease_id(disease_term):
            disease_id = disease_term
        else:
            url_name = f"https://ontology.jax.org/api/network/search/disease?q={quote(disease_term)}&page=0&limit=10"
            response = fetch_data(url_name)
            if response and "results" in response:
                disease_name = response["results"][0]["name"]
                if disease_name.lower() == disease_term.lower():
                    disease_id = response["results"][0]["id"]
        if disease_id is None:
            return "Not Found"
        url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(str(disease_id))}"
        response = fetch_data(url_annotation)
        if response and "categories" in response:
            return response["categories"]
        return "Not Found"
    disease_id = f"OMIM:{disease_term}"
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(disease_id)}"
    response = fetch_data(url_annotation)
    if response and "categories" in response:
        return response["categories"]
    disease_id = f"ORPHA:{disease_term}"
    url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(disease_id)}"
    response = fetch_data(url_annotation)
    if response and "categories" in response:
        return response["categories"]
    return "Not Found"


def run_protein_info(term: str) -> Optional[str]:
    search_url = f"https://rest.uniprot.org/uniprotkb/search?query={term}"
    base_url = "https://www.ebi.ac.uk/proteins/api/proteins"
    search_response = fetch_data(search_url)
    if not search_response:
        print("Error: Unable to fetch data from UniProt.")
        return None

    for entry in search_response.get("results", []):
        primary_accession = entry.get("primaryAccession", "")
        uni_protkb_id = entry.get("uniProtkbId", "")
        alternative_names = [
            name.get("fullName", {}).get("value", "")
            for name in entry.get("proteinDescription", {}).get("alternativeNames", [])
        ]

        if term.lower() in (primary_accession.lower(), *map(str.lower, alternative_names)) or (
            f"{term}_HUMAN".lower() == uni_protkb_id.lower()
        ):
            protein_id = primary_accession
            protein_url = f"{base_url}/{protein_id}"

            protein_response = fetch_data(protein_url)
            if not protein_response:
                print(f"Error: Unable to fetch detailed protein information for '{protein_id}'.")
                return None
            comment_types = {
                "FUNCTION": [],
                "PTM": [],
                "SUBUNIT": [],
                "INTERACTION": [],
                "TISSUE_SPECIFICITY": [],
                "MASS_SPECTROMETRY": [],
                "DISEASE": [],
                "POLYMORPHISM": [],
                "MISCELLANEOUS": [],
                "SIMILARITY": [],
                "CAUTION": [],
            }

            for comment in protein_response.get("comments", []):
                comment_type = comment.get("type")
                if comment_type in comment_types:
                    if comment_type == "INTERACTION":
                        comment_types[comment_type].extend(comment.get("interactions", []))
                    elif comment_type == "MASS_SPECTROMETRY":
                        comment_types[comment_type].append(
                            {
                                "type": comment.get("type"),
                                "molecule": comment.get("molecule"),
                                "method": comment.get("method"),
                                "mass": comment.get("mass"),
                                "error": comment.get("error"),
                            }
                        )
                    elif comment_type == "DISEASE":
                        comment_types[comment_type].append(
                            {
                                "type": comment.get("type"),
                                "diseaseId": comment.get("diseaseId"),
                                "acronym": comment.get("acronym"),
                                "dbReference": comment.get("dbReference"),
                                "description": comment.get("description", {}).get("value", ""),
                            }
                        )
                    else:
                        comment_types[comment_type].extend(
                            text.get("value", "")
                            for text in comment.get("text", [])
                            if isinstance(text, dict)
                        )

            output = {
                "id": protein_response.get("id"),
                "accession": protein_response.get("accession"),
                "secondaryAccession": protein_response.get("secondaryAccession"),
                "protein": protein_response.get("protein"),
                "alternativeName": protein_response.get("alternativeName"),
                "gene": protein_response.get("gene"),
                "comments": comment_types,
                "sequence": protein_response.get("sequence", {}).get("sequence", ""),
            }
            return json.dumps(output, indent=4)

    print(f"No matching protein found for '{term}'.")
    return None


def run_gene_information(search_term: str) -> str:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    summary_url = base_url + "esummary.fcgi"
    gene_headers = ncbi_gene_dataset_headers()
    search_terms = [t.strip() for t in search_term.split(",")]
    gene_information_dict: dict = {}

    def fetch_gene_id(term: str) -> Optional[str]:
        if term.isdigit():
            return term
        url = f"{NCBI_DATASETS_GENE_SYMBOL_BASE}/{term}/taxon/9606"
        response = fetch_data(url, headers=gene_headers)
        if response and "reports" in response:
            return response["reports"][0]["gene"]["gene_id"]
        return None

    def fetch_gene_info(gene_id: str) -> Optional[dict]:
        search_params = {"db": "gene", "term": gene_id, "retmode": "json"}
        search_response = fetch_data(search_url, params=search_params)
        if search_response and "esearchresult" in search_response:
            gene_ids = search_response["esearchresult"].get("idlist", [])
            if gene_ids:
                summary_params = {"db": "gene", "id": ",".join(gene_ids), "retmode": "json"}
                summary_response = fetch_data(summary_url, params=summary_params)
                if summary_response and "result" in summary_response:
                    return summary_response["result"]
        return None

    for term in search_terms:
        gene_id = fetch_gene_id(term)
        if gene_id:
            gene_info = fetch_gene_info(gene_id)
            if gene_info:
                gene_information_dict[term] = gene_info
            else:
                server = "https://grch37.rest.ensembl.org"
                ext = f"/lookup/symbol/homo_sapiens/{term}?"
                response = fetch_data(server + ext, headers={"Content-Type": "application/json"})
                if response:
                    gene_information_dict[term] = response
    if not gene_information_dict:
        return "Not Found"
    return json.dumps(gene_information_dict, indent=4)


def run_disease_information(disease_name: str) -> Union[str, dict]:
    def fetch_bioontology_info(name: str):
        url = f"https://data.bioontology.org/search?q={name}"
        auth = (
            f"apikey token={bioontology_api_key}"
            if bioontology_api_key
            else "apikey token="
        )
        headers = {"Authorization": auth}
        response = fetch_data(url, headers=headers)
        if response:
            collection = response.get("collection", [])
            for item in collection:
                if disease_name.lower() in [item.get("prefLabel", "").lower()] + [
                    syn.lower() for syn in item.get("synonym", [])
                ]:
                    return {
                        "prefLabel": item.get("prefLabel", ""),
                        "synonym": item.get("synonym", []),
                        "definition": item.get("definition", []),
                    }
        return None

    def fetch_orpha_info(name: str):
        url = f"https://api.orphadata.com/rd-cross-referencing/orphacodes/names/{quote(name)}?lang=en"
        response = fetch_data(url)
        if response and "data" in response and "results" in response["data"]:
            data = response["data"]["results"]
            return {
                "ORPHAcode": data.get("ORPHAcode", ""),
                "preferredTerm": data.get("Preferred term", ""),
                "summary": (
                    data.get("SummaryInformation")[0].get("Definition", "")
                    if isinstance(data.get("SummaryInformation"), list)
                    and len(data.get("SummaryInformation")) > 0
                    else ""
                ),
                "Synonym": data.get("Synonym", []),
            }
        return None

    def fetch_hpo_info(name: str):
        url = f"https://ontology.jax.org/api/network/search/disease?q={quote(name)}&page=0&limit=10"
        response = fetch_data(url)
        if response and response.get("results"):
            data = response["results"][0]
            return {
                "id": data.get("id", ""),
                "name": data.get("name", ""),
                "mondoId": data.get("mondoId", ""),
                "description": data.get("description", []),
            }
        return None

    combined_result = {
        "Bioontology": fetch_bioontology_info(disease_name),
        "Orpha": fetch_orpha_info(disease_name),
        "HPO": fetch_hpo_info(disease_name),
    }
    combined_result = {key: value for key, value in combined_result.items() if value}
    return combined_result if combined_result else "Not Found"


def run_disease_genes(disease_term: str) -> Union[str, List[str], None]:
    def is_disease_id(s: str) -> bool:
        return (s.startswith("OMIM:") and s[5:].isdigit()) or (
            s.startswith("ORPHA:") and s[6:].isdigit()
        )

    def fetch_genes_by_id(disease_id: str) -> Optional[List[str]]:
        url_annotation = f"https://ontology.jax.org/api/network/annotation/{quote(disease_id)}"
        response = fetch_data(url_annotation)
        return response.get("genes", []) if response else None

    def fetch_disease_id_by_name(disease_name: str) -> Optional[str]:
        url_name = f"https://ontology.jax.org/api/network/search/disease?q={quote(disease_name)}&page=0&limit=10"
        response = fetch_data(url_name)
        if response and "results" in response and response["results"]:
            for item in response["results"]:
                if item["name"].lower() == disease_name.lower():
                    return item["id"]
        return None

    annotation_result = []
    if disease_term.isdigit():
        disease_id = f"OMIM:{disease_term}"
        genes = fetch_genes_by_id(disease_id)
        if genes:
            return genes
        disease_id = f"ORPHA:{disease_term}"
        genes = fetch_genes_by_id(disease_id)
        if genes:
            return genes
    elif is_disease_id(disease_term):
        genes = fetch_genes_by_id(disease_term)
        if genes:
            return genes
    else:
        disease_id = fetch_disease_id_by_name(disease_term)
        if disease_id:
            genes = fetch_genes_by_id(disease_id)
            if genes:
                return genes

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    summary_url = base_url + "esummary.fcgi"
    search_params = {"db": "omim", "term": disease_term, "retmode": "json"}
    search_response = fetch_data(search_url, params=search_params)
    if (
        search_response
        and "esearchresult" in search_response
        and "idlist" in search_response["esearchresult"]
    ):
        gene_ids = search_response["esearchresult"]["idlist"]
        summary_params = {"db": "omim", "id": ",".join(gene_ids), "retmode": "json"}
        summary_response = fetch_data(summary_url, params=summary_params)
        if (
            summary_response
            and "result" in summary_response
            and "uids" in summary_response["result"]
        ):
            for uid in summary_response["result"]["uids"]:
                title = summary_response["result"][uid]["title"]
                parts = title.split(";")
                if len(parts) > 1:
                    annotation_result.append(parts[-1].strip())
            return annotation_result
    return "Not Found"


def run_snp_information(snp_term: str) -> Union[dict, str, None]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    summary_url = base_url + "esummary.fcgi"
    search_params = {"db": "snp", "term": snp_term, "retmode": "json"}
    search_response = fetch_data(search_url, params=search_params)
    if not search_response:
        return "Request failed or no results found"

    snp_ids = search_response.get("esearchresult", {}).get("idlist", [])
    if not snp_ids:
        print(f"No results found for '{snp_term}'.")
        return None
    summary_params = {"db": "snp", "id": ",".join(snp_ids), "retmode": "json"}
    summary_response = fetch_data(summary_url, params=summary_params)
    if not summary_response:
        print("Failed to retrieve SNP summary information.")
        return None
    return summary_response


def run_sequence_information(sequence_term: str) -> Union[dict, str]:
    sequence = extract_dna_sequence(sequence_term)
    if not sequence:
        return "Not Found"

    try:
        rid = submit_blast(sequence)
        poll_blast_ready(rid)
        blast_json = fetch_blast_json(rid)
        hits = find_hits_recursive(blast_json)
        if not hits:
            return "Not Found"

        top_hit = choose_best_gene_like_hit(hits)
        accession, title = extract_hit_description(top_hit or {})
        gene_symbol = guess_gene_symbol_from_title(title or "")
        aliases = fetch_gene_aliases(gene_symbol) if gene_symbol else []

        return {
            "sequence_length": len(sequence),
            "blast_rid": rid,
            "top_hit": {
                "accession": accession,
                "title": title,
            },
            "gene_symbol_guess": gene_symbol,
            "gene_aliases": aliases,
        }
    except Exception as e:
        return f"Not Found: {e}"


TOOL_REGISTRY_FN: dict[str, Any] = {
    PHENOTYPES_INFO_EXTRACTOR: run_phenotypes_info,
    PHENOTYPES_PARENTS_EXTRACTOR: run_phenotypes_parents,
    PHENOTYPES_CHILDRENS_EXTRACTOR: run_phenotypes_children,
    PHENOTYPES_DISEASE_EXTRACTOR: run_phenotypes_disease,
    PHENOTYPES_GENE_EXTRACTOR: run_phenotypes_gene,
    GENE_PHENOTYPES_EXTRACTOR: run_gene_phenotypes,
    GENE_DISEASES_EXTRACTOR: run_gene_diseases,
    GENE_INFORMATION_TOOL: run_gene_information,
    DISEASE_PHENOTYPES_EXTRACTOR: run_disease_phenotypes,
    DISEASE_INFORMATION_EXTRACTOR: run_disease_information,
    DISEASE_GENE_EXTRACTOR: run_disease_genes,
    PROTEIN_INFORMATION_EXTRACTOR: run_protein_info,
    SNP_INFORMATION_EXTRACTOR: run_snp_information,
    SEQUENCE_INFORMATION_EXTRACTOR: run_sequence_information,
}


def dispatch_by_name(tool_name: str, tool_input: str) -> Any:
    fn = TOOL_REGISTRY_FN.get(tool_name)
    if fn is None:
        raise KeyError(f"Unknown tool: {tool_name!r}. Known: {sorted(TOOL_REGISTRY_FN)}")
    return fn(tool_input)


def format_tool_output(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, (dict, list)):
        return json.dumps(out, ensure_ascii=False, default=str)
    return str(out)


def run_tool(name: str, tool_input: str) -> str:
    return format_tool_output(dispatch_by_name(name, tool_input))
