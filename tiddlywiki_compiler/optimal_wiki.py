#!/usr/bin/env python3
"""
Optimal TiddlyWiki Generator
Uses blockprimes distance-metric theory to create optimally organized wikis
"""

import numpy as np
import json
import re
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
from pathlib import Path
from main import TiddlyWikiCompiler, Tiddler
from compression import DistanceMetricCompressor


class ConceptualSpace:
    """Represents content in a mathematical conceptual space"""
    
    def __init__(self):
        self.concept_vectors = {}
        self.tiddler_positions = {}
        self.distance_matrix = None
        
    def extract_concepts(self, text: str) -> Dict[str, float]:
        """Extract conceptual features from text content"""
        concepts = {}
        
        # Mathematical concepts
        math_patterns = {
            'prime': len(re.findall(r'\bprime\b', text, re.IGNORECASE)),
            'distance': len(re.findall(r'\bdistance\b', text, re.IGNORECASE)),
            'theorem': len(re.findall(r'\btheorem\b', text, re.IGNORECASE)),
            'proof': len(re.findall(r'\bproof\b', text, re.IGNORECASE)),
            'algorithm': len(re.findall(r'\balgorithm\b', text, re.IGNORECASE)),
            'complexity': len(re.findall(r'\bcomplexity\b', text, re.IGNORECASE)),
            'function': len(re.findall(r'\bfunction\b', text, re.IGNORECASE)),
            'sequence': len(re.findall(r'\bsequence\b', text, re.IGNORECASE)),
            'space': len(re.findall(r'\bspace\b', text, re.IGNORECASE)),
            'metric': len(re.findall(r'\bmetric\b', text, re.IGNORECASE)),
        }
        
        # Computational patterns
        comp_patterns = {
            'big_o': len(re.findall(r'O\([^)]+\)', text)),
            'equations': len(re.findall(r'\$[^$]+\$', text)),
            'code_blocks': len(re.findall(r'```[^`]+```', text)),
            'formulas': len(re.findall(r'd_[A-Z]|∑|∏|∫', text)),
            'logic': len(re.findall(r'therefore|thus|hence|implies', text, re.IGNORECASE)),
        }
        
        # Content structure
        structure_patterns = {
            'questions': len(re.findall(r'\?', text)),
            'emphasis': len(re.findall(r'\*\*[^*]+\*\*', text)),
            'lists': len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'headers': len(re.findall(r'^#+\s', text, re.MULTILINE)),
            'length': len(text.split()),
        }
        
        # Combine all features
        concepts.update(math_patterns)
        concepts.update(comp_patterns)
        concepts.update(structure_patterns)
        
        # Normalize by text length
        word_count = max(len(text.split()), 1)
        for key in concepts:
            if key != 'length':
                concepts[key] = concepts[key] / word_count
        
        return concepts
    
    def add_tiddler(self, tiddler: Tiddler):
        """Add a tiddler to the conceptual space"""
        concepts = self.extract_concepts(tiddler.text)
        self.concept_vectors[tiddler.title] = concepts
        
        # Create position vector from concepts
        concept_keys = sorted(concepts.keys())
        position = [concepts.get(key, 0) for key in concept_keys]
        self.tiddler_positions[tiddler.title] = np.array(position)
    
    def compute_distance_matrix(self):
        """Compute distance matrix between all tiddlers"""
        titles = list(self.tiddler_positions.keys())
        n = len(titles)
        self.distance_matrix = np.zeros((n, n))
        
        for i, title_i in enumerate(titles):
            for j, title_j in enumerate(titles):
                if i != j:
                    # Euclidean distance in conceptual space
                    pos_i = self.tiddler_positions[title_i]
                    pos_j = self.tiddler_positions[title_j]
                    distance = np.linalg.norm(pos_i - pos_j)
                    self.distance_matrix[i, j] = distance
        
        return self.distance_matrix, titles
    
    def find_clusters(self, max_distance: float = 0.5) -> List[List[str]]:
        """Find optimal clusters using distance-based grouping"""
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        
        titles = list(self.tiddler_positions.keys())
        clusters = []
        used = set()
        
        for i, title in enumerate(titles):
            if title in used:
                continue
                
            cluster = [title]
            used.add(title)
            
            # Find nearby tiddlers
            for j, other_title in enumerate(titles):
                if other_title not in used and self.distance_matrix[i, j] <= max_distance:
                    cluster.append(other_title)
                    used.add(other_title)
            
            clusters.append(cluster)
        
        return clusters


class OptimalWikiGenerator(TiddlyWikiCompiler):
    """Enhanced compiler that generates optimally organized wikis"""
    
    def __init__(self, source_dir: str, output_file: str = "optimal_wiki.html"):
        super().__init__(source_dir, output_file)
        self.conceptual_space = ConceptualSpace()
        self.optimal_clusters = []
        self.navigation_graph = {}
        self.compression_efficiency = {}
        
    def analyze_content_structure(self):
        """Analyze content using distance-metric theory"""
        print("Analyzing content in conceptual space...")
        
        # Add all tiddlers to conceptual space
        for tiddler in self.tiddlers:
            self.conceptual_space.add_tiddler(tiddler)
        
        # Find optimal clusters
        self.optimal_clusters = self.conceptual_space.find_clusters(max_distance=0.8)
        
        print(f"Found {len(self.optimal_clusters)} optimal clusters:")
        for i, cluster in enumerate(self.optimal_clusters):
            print(f"  Cluster {i+1}: {len(cluster)} tiddlers")
            if len(cluster) <= 3:
                print(f"    {', '.join(cluster)}")
    
    def generate_optimal_tags(self):
        """Generate optimal tag system based on clustering"""
        print("Generating optimal tag system...")
        
        # Add cluster-based tags
        for i, cluster in enumerate(self.optimal_clusters):
            cluster_tag = f"cluster-{i+1}"
            
            # Find dominant concept in cluster
            cluster_concepts = defaultdict(float)
            for title in cluster:
                concepts = self.conceptual_space.concept_vectors.get(title, {})
                for concept, value in concepts.items():
                    cluster_concepts[concept] += value
            
            # Most dominant concept becomes the cluster name
            if cluster_concepts:
                dominant_concept = max(cluster_concepts.items(), key=lambda x: x[1])[0]
                cluster_tag = f"cluster-{dominant_concept}"
            
            # Add cluster tag to all tiddlers in cluster
            for title in cluster:
                for tiddler in self.tiddlers:
                    if tiddler.title == title:
                        if cluster_tag not in tiddler.tags:
                            tiddler.tags.append(cluster_tag)
                        break
    
    def create_navigation_graph(self):
        """Create optimal navigation paths based on distances"""
        print("Creating optimal navigation graph...")
        
        distance_matrix, titles = self.conceptual_space.compute_distance_matrix()
        
        # For each tiddler, find closest neighbors for navigation
        for i, title in enumerate(titles):
            distances = [(distance_matrix[i, j], titles[j]) 
                        for j in range(len(titles)) if i != j]
            distances.sort()
            
            # Take 3 closest neighbors
            closest = [neighbor for _, neighbor in distances[:3]]
            self.navigation_graph[title] = closest
    
    def add_navigation_tiddlers(self):
        """Add navigation helper tiddlers"""
        
        # Create cluster overview tiddler
        cluster_overview = self.create_cluster_overview()
        self.tiddlers.append(cluster_overview)
        
        # Create conceptual map tiddler
        concept_map = self.create_concept_map()
        self.tiddlers.append(concept_map)
        
        # Add navigation links to each tiddler
        for tiddler in self.tiddlers:
            if tiddler.title in self.navigation_graph:
                neighbors = self.navigation_graph[tiddler.title]
                nav_text = "\n\n---\n**Related Topics:** " + " | ".join(
                    f"[[{neighbor}]]" for neighbor in neighbors
                )
                tiddler.text += nav_text
    
    def create_cluster_overview(self) -> Tiddler:
        """Create overview tiddler showing optimal clusters"""
        text = """# Optimal Content Clusters

This wiki has been optimally organized using distance-metric theory. Content is clustered based on conceptual similarity in mathematical space.

"""
        
        for i, cluster in enumerate(self.optimal_clusters):
            text += f"\n## Cluster {i+1}: {len(cluster)} tiddlers\n\n"
            for title in cluster:
                text += f"* [[{title}]]\n"
        
        text += f"\n\n**Total Clusters:** {len(self.optimal_clusters)}\n"
        text += f"**Optimization Method:** Euclidean distance in {len(self.conceptual_space.concept_vectors)} dimensional conceptual space\n"
        
        return Tiddler(
            title="Optimal Clusters",
            text=text,
            tags=["optimization", "clusters", "navigation"],
            fields={"cluster_count": len(self.optimal_clusters)}
        )
    
    def create_concept_map(self) -> Tiddler:
        """Create conceptual map showing mathematical relationships"""
        text = """# Conceptual Distance Map

This map shows the mathematical relationships between content using distance metrics.

## Concept Dimensions

Each tiddler is positioned in multi-dimensional space based on:

"""
        
        # Show the concepts used for positioning
        if self.conceptual_space.concept_vectors:
            sample_concepts = list(self.conceptual_space.concept_vectors.values())[0]
            for concept in sorted(sample_concepts.keys()):
                text += f"* **{concept}**: Frequency/presence in content\n"
        
        text += "\n## Distance Matrix\n\n"
        text += "Pairwise distances between content (smaller = more related):\n\n"
        
        # Show distance matrix for small wikis
        if len(self.tiddlers) <= 10:
            distance_matrix, titles = self.conceptual_space.compute_distance_matrix()
            text += "```\n"
            for i, title_i in enumerate(titles):
                text += f"{title_i[:20]:20s} "
                for j, title_j in enumerate(titles):
                    if i != j:
                        text += f"{distance_matrix[i,j]:.2f} "
                    else:
                        text += "0.00 "
                text += "\n"
            text += "```\n"
        
        return Tiddler(
            title="Concept Map",
            text=text,
            tags=["mathematics", "distance-metric", "analysis"],
            fields={"analysis_type": "conceptual_mapping"}
        )
    
    def optimize_compression(self):
        """Apply optimal compression based on content relationships"""
        print("Applying optimal compression...")
        
        compressor = DistanceMetricCompressor()
        
        # Compress within clusters for better efficiency
        for cluster in self.optimal_clusters:
            cluster_content = []
            for title in cluster:
                for tiddler in self.tiddlers:
                    if tiddler.title == title:
                        cluster_content.append(tiddler.text)
                        break
            
            # Analyze compression efficiency for this cluster
            if cluster_content:
                combined_text = " ".join(cluster_content)
                compressed = compressor.encode_via_distances(combined_text)
                efficiency = len(compressed) / len(combined_text)
                self.compression_efficiency[tuple(cluster)] = efficiency
    
    def compile(self):
        """Enhanced compilation with optimization"""
        print("=== Optimal TiddlyWiki Generation ===")
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_file}")
        print()
        
        # Standard compilation
        super().compile_directory()
        
        # Apply optimizations
        self.analyze_content_structure()
        self.generate_optimal_tags()
        self.create_navigation_graph()
        self.optimize_compression()
        self.add_navigation_tiddlers()
        
        # Generate final wiki
        result = self.generate_wiki_html()
        
        if not result.startswith("Real TiddlyWiki generated:"):
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(result)
        
        print(f"\n=== Optimization Results ===")
        print(f"Total tiddlers: {len(self.tiddlers)}")
        print(f"Optimal clusters: {len(self.optimal_clusters)}")
        print(f"Navigation links: {len(self.navigation_graph)}")
        
        # Show compression efficiency
        if self.compression_efficiency:
            avg_efficiency = np.mean(list(self.compression_efficiency.values()))
            print(f"Average compression efficiency: {avg_efficiency:.2f}")
        
        print(f"Optimized wiki: {self.output_file}")


def generate_optimal_wiki(source_dir: str, output_file: str = "optimal_wiki.html"):
    """Generate an optimally organized TiddlyWiki"""
    generator = OptimalWikiGenerator(source_dir, output_file)
    generator.compile()
    return generator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate optimal TiddlyWiki using distance-metric theory")
    parser.add_argument("source_dir", help="Source directory to compile")
    parser.add_argument("-o", "--output", default="optimal_wiki.html", help="Output wiki file")
    
    args = parser.parse_args()
    
    generate_optimal_wiki(args.source_dir, args.output)