"""
TiddlyWiki HTML Template Generator
Creates a complete TiddlyWiki HTML file with all necessary components
"""

import json
from typing import List, Dict, Any


def generate_tiddlywiki_html(tiddlers_data: List[Dict[str, Any]], title: str = "Compiled Wiki") -> str:
    """Generate a complete TiddlyWiki HTML file"""
    
    # Convert tiddlers to proper TiddlyWiki format
    formatted_tiddlers = []
    for tiddler in tiddlers_data:
        formatted_tiddler = {
            "title": tiddler["title"],
            "text": tiddler["text"],
            "tags": tiddler["tags"],
            "type": tiddler.get("type", "text/vnd.tiddlywiki"),
            "created": tiddler.get("created", ""),
            "modified": tiddler.get("modified", ""),
        }
        
        # Add custom fields
        for key, value in tiddler.items():
            if key not in ["title", "text", "tags", "type", "created", "modified"]:
                formatted_tiddler[key] = value
                
        formatted_tiddlers.append(formatted_tiddler)
    
    # Create tiddlers JSON string
    tiddlers_json = json.dumps(formatted_tiddlers, indent=2, ensure_ascii=False)
    
    html_template = f"""<!doctype html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
<meta name="generator" content="TiddlyWiki Compiler" />
<meta name="tiddlywiki-version" content="5.3.0" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
<meta name="mobile-web-app-capable" content="yes"/>
<meta name="format-detection" content="telephone=no">
<link id="faviconLink" rel="shortcut icon" href="favicon.ico">
<title>{title}</title>

<style>
/* Basic TiddlyWiki CSS */
* {{
    box-sizing: border-box;
}}

body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #333;
    background: #fff;
    margin: 0;
    padding: 0;
}}

.tc-page-container {{
    position: relative;
    min-height: 100vh;
    background: #f4f4f4;
}}

.tc-sidebar-scrollable {{
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 300px;
    background: #f4f4f4;
    border-right: 1px solid #ccc;
    overflow-y: auto;
    padding: 10px;
}}

.tc-story-river {{
    margin-left: 320px;
    padding: 20px;
    background: #fff;
    min-height: 100vh;
}}

.tc-tiddler-frame {{
    margin-bottom: 20px;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.tc-tiddler-title {{
    background: #5778d8;
    color: #fff;
    padding: 10px 15px;
    border-radius: 5px 5px 0 0;
    font-size: 18px;
    font-weight: bold;
}}

.tc-tiddler-body {{
    padding: 15px;
}}

.tc-tiddler-controls {{
    background: #f8f9fa;
    padding: 8px 15px;
    border-top: 1px solid #ddd;
    font-size: 12px;
    color: #666;
}}

.tc-tag {{
    display: inline-block;
    background: #e7f3ff;
    color: #0366d6;
    padding: 2px 8px;
    margin: 2px;
    border-radius: 12px;
    font-size: 11px;
    text-decoration: none;
}}

.tc-tag:hover {{
    background: #cce7ff;
}}

.tc-tag.tag-proof {{
    background: #fff3cd;
    color: #856404;
}}

.tc-tag.tag-demo {{
    background: #d4edda;
    color: #155724;
}}

.tc-tag.tag-question {{
    background: #f8d7da;
    color: #721c24;
}}

.tc-tag.tag-follow-up {{
    background: #e2e3e5;
    color: #383d41;
}}

.tc-sidebar-tab {{
    display: block;
    padding: 8px 12px;
    margin: 2px 0;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 3px;
    text-decoration: none;
    color: #333;
    transition: background 0.2s;
}}

.tc-sidebar-tab:hover {{
    background: #e9ecef;
}}

.tc-sidebar-tab.tc-tab-selected {{
    background: #5778d8;
    color: #fff;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #333;
    margin-top: 20px;
    margin-bottom: 10px;
}}

h1 {{ font-size: 24px; }}
h2 {{ font-size: 20px; }}
h3 {{ font-size: 16px; }}

pre {{
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 3px;
    padding: 10px;
    overflow-x: auto;
}}

code {{
    background: #f8f9fa;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
}}

.tc-search-box {{
    width: 100%;
    padding: 8px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
    border-radius: 3px;
}}

.mathematical {{
    font-style: italic;
}}

.tc-summary {{
    background: #e7f3ff;
    padding: 10px;
    border-left: 4px solid #0366d6;
    margin-bottom: 15px;
}}

.tc-key-concepts {{
    background: #f8f9fa;
    padding: 8px;
    border-radius: 3px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #666;
}}
</style>

<script>
window.$tw = window.$tw || {{}};
$tw.boot = $tw.boot || {{}};
$tw.utils = $tw.utils || {{}};

// Preload tiddlers
$tw.preloadTiddlers = {tiddlers_json};

// Basic TiddlyWiki functionality
$tw.utils.parseStringArray = function(value) {{
    if(typeof value === "string") {{
        return value.split(" ").filter(function(item) {{ return item.length > 0; }});
    }}
    return [];
}};

$tw.utils.stringifyList = function(value) {{
    return value.join(" ");
}};

// Simple wiki rendering
$tw.wiki = {{
    getTiddler: function(title) {{
        return $tw.preloadTiddlers.find(function(tiddler) {{
            return tiddler.title === title;
        }});
    }},
    
    getTiddlers: function() {{
        return $tw.preloadTiddlers.map(function(tiddler) {{
            return tiddler.title;
        }});
    }},
    
    filterTiddlers: function(filter) {{
        // Simple tag-based filtering
        if(filter.startsWith("[tag[") && filter.endsWith("]]")) {{
            var tag = filter.slice(5, -2);
            return $tw.preloadTiddlers.filter(function(tiddler) {{
                var tags = $tw.utils.parseStringArray(tiddler.tags);
                return tags.indexOf(tag) !== -1;
            }}).map(function(tiddler) {{
                return tiddler.title;
            }});
        }}
        return $tw.preloadTiddlers.map(function(tiddler) {{
            return tiddler.title;
        }});
    }}
}};

// Render tiddler content
function renderTiddler(title) {{
    var tiddler = $tw.wiki.getTiddler(title);
    if(!tiddler) return "";
    
    var text = tiddler.text || "";
    var tags = $tw.utils.parseStringArray(tiddler.tags);
    
    // Simple markdown-like rendering
    text = text.replace(/^!! (.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^! (.+)$/gm, '<h1>$1</h1>');
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/\\n/g, '<br>');
    
    // Render special sections
    if(text.includes('!! Summary')) {{
        text = text.replace(/!! Summary\\s*([^!]+)/g, '<div class="tc-summary">$1</div>');
    }}
    
    if(text.includes('!! Key Concepts')) {{
        text = text.replace(/!! Key Concepts\\s*([^!]+)/g, '<div class="tc-key-concepts"><strong>Key Concepts:</strong> $1</div>');
    }}
    
    var tagHtml = tags.map(function(tag) {{
        return '<a href="#" class="tc-tag tag-' + tag.replace(/[^a-zA-Z0-9]/g, '-') + '" onclick="filterByTag(\\''+tag+'\\'); return false;">' + tag + '</a>';
    }}).join(' ');
    
    return '<div class="tc-tiddler-frame">' +
           '<div class="tc-tiddler-title">' + title + '</div>' +
           '<div class="tc-tiddler-body">' + text + '</div>' +
           '<div class="tc-tiddler-controls">Tags: ' + tagHtml + '</div>' +
           '</div>';
}}

// Render all tiddlers
function renderStory() {{
    var storyElement = document.getElementById('tc-story-river');
    var html = '';
    
    $tw.wiki.getTiddlers().forEach(function(title) {{
        html += renderTiddler(title);
    }});
    
    storyElement.innerHTML = html;
}}

// Filter by tag
function filterByTag(tag) {{
    var storyElement = document.getElementById('tc-story-river');
    var html = '';
    
    var filteredTiddlers = $tw.wiki.filterTiddlers('[tag[' + tag + ']]');
    filteredTiddlers.forEach(function(title) {{
        html += renderTiddler(title);
    }});
    
    storyElement.innerHTML = html;
}}

// Search functionality
function searchTiddlers() {{
    var query = document.getElementById('search-input').value.toLowerCase();
    var storyElement = document.getElementById('tc-story-river');
    var html = '';
    
    $tw.preloadTiddlers.forEach(function(tiddler) {{
        if(tiddler.title.toLowerCase().includes(query) || 
           (tiddler.text && tiddler.text.toLowerCase().includes(query))) {{
            html += renderTiddler(tiddler.title);
        }}
    }});
    
    storyElement.innerHTML = html;
}}

// Get unique tags
function getUniqueTags() {{
    var allTags = [];
    $tw.preloadTiddlers.forEach(function(tiddler) {{
        var tags = $tw.utils.parseStringArray(tiddler.tags);
        allTags = allTags.concat(tags);
    }});
    
    return [...new Set(allTags)].sort();
}}

// Initialize the wiki
window.addEventListener('load', function() {{
    renderStory();
    
    // Populate sidebar with tags
    var tagsElement = document.getElementById('tags-list');
    var tags = getUniqueTags();
    
    tags.forEach(function(tag) {{
        var tagElement = document.createElement('a');
        tagElement.href = '#';
        tagElement.className = 'tc-sidebar-tab';
        tagElement.textContent = tag;
        tagElement.onclick = function() {{ filterByTag(tag); return false; }};
        tagsElement.appendChild(tagElement);
    }});
}});
</script>
</head>

<body class="tc-body">
<div class="tc-page-container">
    <div class="tc-sidebar-scrollable">
        <h3>Search</h3>
        <input type="text" id="search-input" class="tc-search-box" placeholder="Search tiddlers..." onkeyup="searchTiddlers()">
        
        <h3>Navigation</h3>
        <a href="#" class="tc-sidebar-tab" onclick="renderStory(); return false;">All Tiddlers</a>
        
        <h3>Tags</h3>
        <div id="tags-list"></div>
        
        <h3>Statistics</h3>
        <div style="font-size: 12px; color: #666; padding: 8px;">
            <div>Total Tiddlers: <span id="tiddler-count">{len(formatted_tiddlers)}</span></div>
            <div>Total Tags: <span id="tag-count"></span></div>
        </div>
    </div>
    
    <div class="tc-story-river" id="tc-story-river">
        <!-- Tiddlers will be rendered here -->
    </div>
</div>

<script>
// Update tag count after initialization
window.addEventListener('load', function() {{
    document.getElementById('tag-count').textContent = getUniqueTags().length;
}});
</script>

</body>
</html>"""
    
    return html_template