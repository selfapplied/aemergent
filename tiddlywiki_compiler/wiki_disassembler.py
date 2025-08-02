#!/usr/bin/env python3
"""
TiddlyWiki Disassembler
Extracts, analyzes, and reverse-engineers TiddlyWiki structure using distance-metric theory
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import argparse

from compression import DistanceMetricCompressor


class WikiStructureAnalyzer:
    """Analyzes the mathematical structure of extracted wiki content"""
    
    def __init__(self):
        self.tiddlers = {}
        self.concept_analysis = {}
        self.cluster_analysis = {}
        self.compression_analysis = {}
        self.navigation_patterns = {}
        
    def analyze_tiddler_relationships(self) -> Dict[str, Any]:
        """Analyze mathematical relationships between tiddlers"""
        
        # Extract conceptual features from each tiddler
        concept_vectors = {}
        for title, content in self.tiddlers.items():
            concept_vectors[title] = self.extract_mathematical_features(content)
        
        # Compute distance matrix
        titles = list(concept_vectors.keys())
        n = len(titles)
        distance_matrix = np.zeros((n, n))
        
        for i, title_i in enumerate(titles):
            for j, title_j in enumerate(titles):
                if i != j:
                    vector_i = np.array(list(concept_vectors[title_i].values()))
                    vector_j = np.array(list(concept_vectors[title_j].values()))
                    distance = np.linalg.norm(vector_i - vector_j)
                    distance_matrix[i, j] = distance
        
        return {
            "concept_vectors": concept_vectors,
            "distance_matrix": distance_matrix.tolist(),
            "titles": titles,
            "dimensions": len(list(concept_vectors.values())[0]) if concept_vectors else 0
        }
    
    def extract_mathematical_features(self, text: str) -> Dict[str, float]:
        """Extract mathematical features from text (same as optimal_wiki.py)"""
        features = {}
        
        # Mathematical concepts
        math_patterns = {
            'prime_density': len(re.findall(r'\bprime\b', text, re.IGNORECASE)),
            'distance_refs': len(re.findall(r'\bdistance\b', text, re.IGNORECASE)),
            'theorem_count': len(re.findall(r'\btheorem\b', text, re.IGNORECASE)),
            'proof_structures': len(re.findall(r'\bproof\b', text, re.IGNORECASE)),
            'algorithm_content': len(re.findall(r'\balgorithm\b', text, re.IGNORECASE)),
            'complexity_analysis': len(re.findall(r'\bcomplexity\b', text, re.IGNORECASE)),
        }
        
        # Computational patterns
        comp_patterns = {
            'big_o_notation': len(re.findall(r'O\([^)]+\)', text)),
            'mathematical_formulas': len(re.findall(r'\$[^$]+\$|d_[A-Z]|∑|∏|∫', text)),
            'code_density': len(re.findall(r'```[^`]+```', text)),
            'logical_connectives': len(re.findall(r'therefore|thus|hence|implies', text, re.IGNORECASE)),
        }
        
        # Structural features
        structure_patterns = {
            'question_density': len(re.findall(r'\?', text)),
            'emphasis_markers': len(re.findall(r'\*\*[^*]+\*\*', text)),
            'list_structures': len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'header_hierarchy': len(re.findall(r'^#+\s', text, re.MULTILINE)),
        }
        
        # Combine and normalize
        features.update(math_patterns)
        features.update(comp_patterns)
        features.update(structure_patterns)
        
        # Normalize by word count
        word_count = max(len(text.split()), 1)
        for key in features:
            features[key] = features[key] / word_count
        
        return features
    
    def detect_clusters(self, distance_matrix: List[List[float]], titles: List[str], 
                       threshold: float = 0.5) -> List[List[str]]:
        """Detect clusters from distance matrix"""
        n = len(titles)
        clusters = []
        used = set()
        
        for i in range(n):
            if titles[i] in used:
                continue
                
            cluster = [titles[i]]
            used.add(titles[i])
            
            for j in range(n):
                if i != j and titles[j] not in used and distance_matrix[i][j] <= threshold:
                    cluster.append(titles[j])
                    used.add(titles[j])
            
            clusters.append(cluster)
        
        return clusters
    
    def analyze_compression_patterns(self) -> Dict[str, Any]:
        """Analyze compression patterns in the content"""
        compressor = DistanceMetricCompressor()
        compression_data = {}
        
        for title, content in self.tiddlers.items():
            # Analyze distance-based compression potential
            distances = compressor.compute_pattern_distance(content)
            compressed = compressor.encode_via_distances(content)
            
            compression_data[title] = {
                "original_length": len(content),
                "compressed_length": len(compressed),
                "compression_ratio": len(compressed) / len(content) if content else 1.0,
                "pattern_distances": distances,
                "compression_method": "distance_metric"
            }
        
        return compression_data
    
    def reverse_engineer_navigation(self) -> Dict[str, List[str]]:
        """Reverse-engineer navigation patterns from wiki links"""
        navigation_graph = defaultdict(list)
        
        for title, content in self.tiddlers.items():
            # Find wiki links [[target]]
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            
            # Find related topics sections
            related_section = re.search(r'\*\*Related Topics:\*\*(.+?)(?:\n|$)', content, re.IGNORECASE)
            if related_section:
                related_links = re.findall(r'\[\[([^\]]+)\]\]', related_section.group(1))
                links.extend(related_links)
            
            navigation_graph[title] = list(set(links))  # Remove duplicates
        
        return dict(navigation_graph)


class TiddlyWikiDisassembler:
    """Main disassembler class"""
    
    def __init__(self, wiki_file: str):
        self.wiki_file = Path(wiki_file)
        self.tiddlers = {}
        self.metadata = {}
        self.structure_analyzer = WikiStructureAnalyzer()
        
    def extract_tiddlers(self) -> Dict[str, Dict[str, Any]]:
        """Extract tiddlers from TiddlyWiki HTML file"""
        
        with open(self.wiki_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to find tiddlers in the HTML
        tiddlers = {}
        
        # Look for JSON data in script tags
        json_match = re.search(r'preloadTiddlers:\s*(\[.*?\])', content, re.DOTALL)
        if json_match:
            try:
                tiddlers_data = json.loads(json_match.group(1))
                for tiddler in tiddlers_data:
                    title = tiddler.get('title', 'Untitled')
                    tiddlers[title] = tiddler
            except json.JSONDecodeError:
                print("Warning: Could not parse tiddlers JSON data")
        
        # Alternative: look for individual tiddler data
        if not tiddlers:
            tiddler_matches = re.findall(r'<div[^>]*title="([^"]*)"[^>]*>(.*?)</div>', content, re.DOTALL)
            for title, text in tiddler_matches:
                if title and text.strip():
                    tiddlers[title] = {
                        'title': title,
                        'text': text.strip(),
                        'tags': '',
                        'type': 'text/vnd.tiddlywiki'
                    }
        
        self.tiddlers = tiddlers
        return tiddlers
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Perform comprehensive structural analysis"""
        
        # Prepare data for analysis
        self.structure_analyzer.tiddlers = {
            title: data.get('text', '') for title, data in self.tiddlers.items()
        }
        
        # Perform analyses
        relationship_analysis = self.structure_analyzer.analyze_tiddler_relationships()
        compression_analysis = self.structure_analyzer.analyze_compression_patterns()
        navigation_analysis = self.structure_analyzer.reverse_engineer_navigation()
        
        # Detect clusters
        if relationship_analysis.get('distance_matrix'):
            clusters = self.structure_analyzer.detect_clusters(
                relationship_analysis['distance_matrix'],
                relationship_analysis['titles']
            )
        else:
            clusters = []
        
        return {
            "total_tiddlers": len(self.tiddlers),
            "relationships": relationship_analysis,
            "clusters": clusters,
            "compression": compression_analysis,
            "navigation": navigation_analysis,
            "optimization_potential": self.assess_optimization_potential(relationship_analysis, clusters)
        }
    
    def assess_optimization_potential(self, relationships: Dict, clusters: List[List[str]]) -> Dict[str, Any]:
        """Assess how well the wiki is optimized"""
        
        if not relationships.get('distance_matrix'):
            return {"assessment": "insufficient_data"}
        
        distance_matrix = np.array(relationships['distance_matrix'])
        n = len(relationships['titles'])
        
        # Calculate average intra-cluster vs inter-cluster distances
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        title_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for title in cluster:
                title_to_cluster[title] = i
        
        for i in range(n):
            for j in range(i+1, n):
                title_i = relationships['titles'][i]
                title_j = relationships['titles'][j]
                distance = distance_matrix[i][j]
                
                if title_to_cluster.get(title_i) == title_to_cluster.get(title_j):
                    intra_cluster_distances.append(distance)
                else:
                    inter_cluster_distances.append(distance)
        
        avg_intra = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        avg_inter = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        # Good clustering should have small intra-cluster and large inter-cluster distances
        optimization_score = (avg_inter - avg_intra) / (avg_inter + avg_intra) if (avg_inter + avg_intra) > 0 else 0
        
        return {
            "optimization_score": optimization_score,
            "avg_intra_cluster_distance": avg_intra,
            "avg_inter_cluster_distance": avg_inter,
            "cluster_count": len(clusters),
            "assessment": "well_optimized" if optimization_score > 0.2 else "needs_optimization"
        }
    
    def export_tiddlers(self, output_dir: str, format: str = "markdown"):
        """Export tiddlers back to individual files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for title, data in self.tiddlers.items():
            # Sanitize filename
            filename = re.sub(r'[^\w\s-]', '_', title)
            filename = re.sub(r'[-\s]+', '_', filename)
            
            if format == "markdown":
                file_path = output_path / f"{filename}.md"
                content = self.convert_to_markdown(data)
            elif format == "json":
                file_path = output_path / f"{filename}.json"
                content = json.dumps(data, indent=2)
            else:
                file_path = output_path / f"{filename}.txt"
                content = data.get('text', '')
            
            file_path.write_text(content, encoding='utf-8')
        
        print(f"Exported {len(self.tiddlers)} tiddlers to {output_path}")
    
    def convert_to_markdown(self, tiddler_data: Dict[str, Any]) -> str:
        """Convert tiddler to markdown format"""
        content = f"# {tiddler_data.get('title', 'Untitled')}\n\n"
        
        # Add metadata
        tags = tiddler_data.get('tags', '')
        if tags:
            content += f"**Tags:** {tags}\n\n"
        
        # Add main content
        text = tiddler_data.get('text', '')
        content += text
        
        # Add metadata footer
        content += "\n\n---\n"
        content += f"**Type:** {tiddler_data.get('type', 'unknown')}\n"
        content += f"**Created:** {tiddler_data.get('created', 'unknown')}\n"
        content += f"**Modified:** {tiddler_data.get('modified', 'unknown')}\n"
        
        return content
    
    def generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"""# TiddlyWiki Structure Analysis Report

## Overview
- **Total Tiddlers:** {analysis['total_tiddlers']}
- **Conceptual Dimensions:** {analysis['relationships'].get('dimensions', 0)}
- **Detected Clusters:** {len(analysis['clusters'])}

## Optimization Assessment
"""
        
        optimization = analysis.get('optimization_potential', {})
        if optimization.get('optimization_score') is not None:
            score = optimization['optimization_score']
            report += f"- **Optimization Score:** {score:.3f} ({optimization['assessment']})\n"
            report += f"- **Avg Intra-cluster Distance:** {optimization['avg_intra_cluster_distance']:.3f}\n"
            report += f"- **Avg Inter-cluster Distance:** {optimization['avg_inter_cluster_distance']:.3f}\n"
        
        report += "\n## Cluster Analysis\n\n"
        for i, cluster in enumerate(analysis['clusters']):
            report += f"### Cluster {i+1} ({len(cluster)} tiddlers)\n"
            for title in cluster:
                report += f"- {title}\n"
            report += "\n"
        
        report += "## Compression Analysis\n\n"
        compression_data = analysis.get('compression', {})
        if compression_data:
            total_original = sum(data['original_length'] for data in compression_data.values())
            total_compressed = sum(data['compressed_length'] for data in compression_data.values())
            avg_ratio = total_compressed / total_original if total_original > 0 else 1.0
            
            report += f"- **Average Compression Ratio:** {avg_ratio:.3f}\n"
            report += f"- **Total Space Savings:** {((1 - avg_ratio) * 100):.1f}%\n\n"
        
        report += "## Navigation Patterns\n\n"
        navigation = analysis.get('navigation', {})
        if navigation:
            total_links = sum(len(links) for links in navigation.values())
            avg_links = total_links / len(navigation) if navigation else 0
            report += f"- **Total Navigation Links:** {total_links}\n"
            report += f"- **Average Links per Tiddler:** {avg_links:.1f}\n"
        
        return report
    
    def disassemble(self, output_dir: Optional[str] = None, export_format: str = "markdown") -> Dict[str, Any]:
        """Main disassembly method"""
        
        print(f"Disassembling TiddlyWiki: {self.wiki_file}")
        
        # Extract tiddlers
        tiddlers = self.extract_tiddlers()
        print(f"Extracted {len(tiddlers)} tiddlers")
        
        # Analyze structure
        analysis = self.analyze_structure()
        print(f"Detected {len(analysis['clusters'])} content clusters")
        
        # Export if requested
        if output_dir:
            self.export_tiddlers(output_dir, export_format)
        
        # Generate report
        report = self.generate_analysis_report(analysis)
        
        return {
            "tiddlers": tiddlers,
            "analysis": analysis,
            "report": report
        }


def main():
    parser = argparse.ArgumentParser(description="Disassemble and analyze TiddlyWiki files")
    parser.add_argument("wiki_file", help="TiddlyWiki HTML file to disassemble")
    parser.add_argument("-o", "--output", help="Output directory for extracted tiddlers")
    parser.add_argument("-f", "--format", choices=["markdown", "json", "text"], 
                       default="markdown", help="Export format")
    parser.add_argument("-r", "--report", help="Output file for analysis report")
    
    args = parser.parse_args()
    
    # Disassemble the wiki
    disassembler = TiddlyWikiDisassembler(args.wiki_file)
    results = disassembler.disassemble(args.output, args.format)
    
    # Save analysis report
    if args.report:
        with open(args.report, 'w') as f:
            f.write(results['report'])
        print(f"Analysis report saved to {args.report}")
    else:
        print("\n" + results['report'])


if __name__ == "__main__":
    main()