#!/usr/bin/env python3
"""
角色状态可视化 HTML 生成器（支持按 version 筛选）
读取 CSV 文件并生成包含三个 Tab 视图的独立 HTML 文件
"""

import csv
import json
from collections import defaultdict
import re


def extract_scene_number(title):
    """从 title 中提取场次数字"""
    match = re.search(r'(\d+)', title)
    return int(match.group(1)) if match else 999999


def extract_subtitle_number(subtitle):
    """从 subtitle 中提取数字"""
    if not subtitle:
        return 0
    match = re.search(r'(\d+)', subtitle)
    return int(match.group(1)) if match else 0


def load_and_process_csv(csv_file):
    """加载并处理 CSV 数据（带 version）"""
    scenes = {}
    characters_set = set()
    character_timelines = defaultdict(list)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row['title']
            subtitle = row['subtitle']
            character = row['character']
            version = (row.get('version') or '').strip()

            full_title = f"{title} {subtitle}" if subtitle else title
            # 用 version + full_title 区分不同 Part 的同名场次
            scene_key = f"{version}__{full_title}"
            
            # 处理场次数据
            if scene_key not in scenes:
                scenes[scene_key] = {
                    'sceneNumber': extract_scene_number(title),
                    'title': title,
                    'subtitle': subtitle,
                    'fullTitle': full_title,
                    'sceneCategory': row.get('scene_category', ''),
                    'lighting': row.get('lighting', ''),
                    'space': row.get('space', ''),
                    'region': row.get('region', ''),
                    'mainLocation': row.get('main_location', ''),
                    'subLocation': row.get('sub_location', ''),
                    'summary': row.get('summary', ''),
                    'version': version,
                    'characters': []
                }
            
            # 添加角色
            if character:
                characters_set.add(character)
                scenes[scene_key]['characters'].append({
                    'name': character,
                    'status': row.get('status', '')
                })
                
                # 构建角色时间线（带 version）
                character_timelines[character].append({
                    'sceneNumber': extract_scene_number(title),
                    'fullTitle': full_title,
                    'status': row.get('status', ''),
                    'summary': row.get('summary', ''),
                    'version': version,
                })
    
    # 转换为列表并排序（按 version + 场次号）
    scenes_list = list(scenes.values())
    scenes_list.sort(
        key=lambda s: (
            s.get('version', ''),
            s['sceneNumber'],
            extract_subtitle_number(s['subtitle']),
            s['subtitle'],
        )
    )
    
    # 排序角色时间线
    for character in character_timelines:
        character_timelines[character].sort(
            key=lambda s: (s.get('version', ''), s['sceneNumber'])
        )
    
    characters_list = sorted(list(characters_set))
    
    # 按出场次数排序角色
    timelines_list = [
        {
            'character': char,
            'scenes': character_timelines[char],
            'count': len(character_timelines[char])
        }
        for char in characters_list
    ]
    timelines_list.sort(key=lambda t: t['count'], reverse=True)
    
    return scenes_list, characters_list, timelines_list


def generate_scene_card(scene):
    """生成单个场次卡片的 HTML（增加 data-version）"""
    characters_html = ''
    if scene['characters']:
        characters_html = '<div class="characters-grid">' + ''.join(
            f'''<div class="character-box">
                <h4>{char['name']}</h4>
                <p>{char['status']}</p>
            </div>'''
            for char in scene['characters']
        ) + '</div>'
    else:
        characters_html = '<div class="no-characters">该场次无主要角色</div>'
    
    return f'''
    <div class="scene-card" data-version="{scene.get('version', '')}">
        <h2>{scene['fullTitle']}</h2>
        <div class="scene-info">
            <div><span>场景类型：</span> {scene['sceneCategory']}</div>
            <div><span>光线：</span> {scene['lighting']}</div>
            <div><span>空间：</span> {scene['space']}</div>
            <div><span>地区：</span> {scene['region']}</div>
            <div style="grid-column: 1 / -1;"><span>地点：</span> {scene['mainLocation']}{' - ' + scene['subLocation'] if scene['subLocation'] else ''}</div>
        </div>
        <div class="scene-summary">
            <h4>剧情概要</h4>
            <p>{scene['summary']}</p>
        </div>
        <div>
            <h4 style="margin-bottom: 1rem;">出场角色</h4>
            {characters_html}
        </div>
    </div>
    '''


def generate_html(scenes, characters, timelines, output_file):
    """生成完整的 HTML 文件（支持 version 筛选）"""

    # 从 scenes 里收集所有非空 version
    all_versions = sorted(
        {(s.get('version') or '').strip() for s in scenes if (s.get('version') or '').strip()}
    )
    version_options_html = '<option value="all">全部版本</option>' + ''.join(
        f'<option value="{v}">{v}</option>' for v in all_versions
    )
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 角色状态可视化</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 2rem;
        }}
        
        .fixed-header {{
            flex-shrink: 0;
        }}
        
        .scrollable-content {{
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 0.5rem;
        }}
        
        .scrollable-content::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .scrollable-content::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }}
        
        .scrollable-content::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }}
        
        .scrollable-content::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            color: #9ca3af;
            font-size: 0.9rem;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        
        .stat-card h3 {{
            font-size: 0.875rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        
        .stat-card:nth-child(1) .value {{ color: #00d4ff; }}
        .stat-card:nth-child(2) .value {{ color: #a855f7; }}
        .stat-card:nth-child(3) .value {{ color: #fbbf24; }}
        
        .version-filter {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            color: #9ca3af;
            font-size: 0.9rem;
        }}
        
        .version-filter select {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            color: #e0e0e0;
            padding: 0.5rem 0.75rem;
        }}
        
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }}
        
        .tab {{
            padding: 1rem 2rem;
            background: transparent;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        
        .tab:hover {{
            color: #e0e0e0;
        }}
        
        .tab.active {{
            color: #00d4ff;
            border-bottom-color: #00d4ff;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* Timeline View */
        .timeline-controls {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .search-box {{
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1rem;
            margin-bottom: 1rem;
        }}
        
        .character-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }}
        
        .chip {{
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.875rem;
        }}
        
        .chip:hover {{
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
        }}
        
        .chip.selected {{
            background: #00d4ff;
            color: #000;
            border-color: #00d4ff;
        }}
        
        .clear-btn {{
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #e0e0e0;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .clear-btn:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .timeline-grid {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            overflow-x: auto;
        }}
        
        .timeline-table {{
            min-width: max-content;
        }}
        
        .timeline-header {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .timeline-row {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            align-items: center;
        }}
        
        .character-label {{
            width: 120px;
            flex-shrink: 0;
            font-size: 0.875rem;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .scene-label {{
            width: 60px;
            flex-shrink: 0;
            text-align: center;
            font-size: 0.75rem;
            color: #00d4ff;
        }}
        
        .timeline-cell {{
            width: 60px;
            height: 40px;
            flex-shrink: 0;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .timeline-cell.has-character {{
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
        }}
        
        .timeline-cell.has-character:hover {{
            background: rgba(0, 212, 255, 0.3);
            transform: scale(1.05);
        }}
        
        .timeline-cell.has-character .dot {{
            width: 12px;
            height: 12px;
            background: #00d4ff;
            border-radius: 50%;
        }}
        
        /* Scenes View */
        .scene-card {{
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #00d4ff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s;
        }}
        
        .scene-card:hover {{
            box-shadow: 0 8px 24px rgba(0, 212, 255, 0.2);
        }}
        
        .scene-card h2 {{
            color: #00d4ff;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }}
        
        .scene-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.875rem;
        }}
        
        .scene-info span {{
            color: #9ca3af;
        }}
        
        .scene-summary {{
            background: rgba(251, 191, 36, 0.1);
            border-left: 3px solid #fbbf24;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        .scene-summary h4 {{
            color: #fbbf24;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}
        
        .scene-summary p {{
            line-height: 1.6;
            font-size: 0.875rem;
        }}
        
        .characters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }}
        
        .character-box {{
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            padding: 1rem;
            transition: all 0.3s;
        }}
        
        .character-box:hover {{
            background: rgba(0, 212, 255, 0.2);
        }}
        
        .character-box h4 {{
            color: #00d4ff;
            margin-bottom: 0.5rem;
        }}
        
        .character-box p {{
            font-size: 0.875rem;
            line-height: 1.5;
        }}
        
        .no-characters {{
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            color: #9ca3af;
        }}
        
        /* Characters View */
        .character-selector {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .character-selector label {{
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        
        .character-selector select {{
            width: 100%;
            max-width: 400px;
            padding: 0.75rem 1rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1rem;
        }}
        
        .timeline-card {{
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #a855f7;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .timeline-card h3 {{
            color: #a855f7;
            font-size: 1.125rem;
            margin-bottom: 1rem;
        }}
        
        .status-box {{
            background: rgba(0, 212, 255, 0.1);
            border-left: 3px solid #00d4ff;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        .status-box h4 {{
            color: #00d4ff;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}
        
        .status-box p {{
            line-height: 1.6;
            font-size: 0.875rem;
        }}
        
        .plot-box {{
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
        }}
        
        .plot-box h4 {{
            color: #9ca3af;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}
        
        .plot-box p {{
            line-height: 1.6;
            font-size: 0.875rem;
            color: #d1d5db;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: #9ca3af;
        }}
        
        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        
        .modal.show {{
            display: flex;
        }}
        
        .modal-content {{
            background: #1a1a2e;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 1.5rem;
        }}
        
        .modal-header h2 {{
            color: #00d4ff;
            font-size: 1.25rem;
        }}
        
        .close-btn {{
            background: none;
            border: none;
            color: #9ca3af;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
        }}
        
        .close-btn:hover {{
            color: #e0e0e0;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            .tabs {{
                flex-wrap: wrap;
            }}
            .version-filter {{
                flex-wrap: wrap;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="fixed-header">
            <div class="header">
                <h1>🎬 角色状态可视化</h1>
                <p>Character Status Visualization</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>总场次数</h3>
                    <div class="value">{len(scenes)}</div>
                </div>
                <div class="stat-card">
                    <h3>角色数量</h3>
                    <div class="value">{len(characters)}</div>
                </div>
                <div class="stat-card">
                    <h3>平均角色/场次</h3>
                    <div class="value">{sum(len(s['characters']) for s in scenes) / len(scenes):.1f}</div>
                </div>
            </div>

            <div class="version-filter">
                <span>按版本筛选：</span>
                <select id="version-select" onchange="onVersionChange()">
                    {version_options_html}
                </select>
                <span id="version-hint"></span>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab('timeline', event)">时间线泳道图</button>
                <button class="tab" onclick="switchTab('scenes', event)">按场次查看</button>
                <button class="tab" onclick="switchTab('characters', event)">按角色查看</button>
            </div>
        </div>
        
        <div class="scrollable-content">
            <!-- Timeline View -->
            <div id="timeline-view" class="tab-content active">
                <div class="timeline-controls">
                    <input type="text" id="search-input" class="search-box" placeholder="搜索角色..." onkeyup="filterCharacters()">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div id="selected-info" style="font-size: 0.875rem; color: #9ca3af;"></div>
                        <button id="clear-btn" class="clear-btn" onclick="clearSelection()" style="display: none;">清除选择</button>
                    </div>
                    <div id="character-chips" class="character-chips"></div>
                </div>
                
                <div class="timeline-grid">
                    <div class="timeline-table">
                        <div class="timeline-header" id="timeline-header">
                            <div class="character-label" style="color: #9ca3af;">角色 / 场次</div>
                        </div>
                        <div id="timeline-rows"></div>
                    </div>
                </div>
            </div>
        
            <!-- Scenes View -->
            <div id="scenes-view" class="tab-content">
                {''.join(generate_scene_card(s) for s in scenes)}
            </div>
        
            <!-- Characters View -->
            <div id="characters-view" class="tab-content">
                <div class="character-selector">
                    <label>选择角色:</label>
                    <select id="character-select" onchange="showCharacterTimeline()">
                        <option value="">-- 请选择角色 --</option>
                        {''.join(f'<option value="{{i}}">{{t["character"]}} ({{t["count"]}})</option>' for i, t in enumerate(timelines))}
                    </select>
                </div>
                <div id="character-timeline"></div>
            </div>
        </div>
    </div>
    
    <!-- Modal -->
    <div id="modal" class="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2 id="modal-title"></h2>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div id="modal-body"></div>
        </div>
    </div>
    
    <script>
        const allScenes = {json.dumps(scenes, ensure_ascii=False)};
        const characters = {json.dumps(characters, ensure_ascii=False)};
        const timelines = {json.dumps(timelines, ensure_ascii=False)};
        
        let selectedCharacters = new Set();
        let filteredCharacters = [...characters];
        let showAll = false;
        let currentVersion = 'all';
        let filteredScenes = allScenes.slice();
        
        // Initialize
        renderCharacterChips();
        renderTimelineHeader();
        renderTimelineRows();
        applyVersionFilterToScenes();
        showCharacterTimeline();
        const versionHintEl = document.getElementById('version-hint');
        if (versionHintEl) {{
            versionHintEl.textContent = '当前显示：全部版本';
        }}
        
        function switchTab(tabName, evt) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (evt && evt.target) {{
                evt.target.classList.add('active');
            }}
            const view = document.getElementById(tabName + '-view');
            if (view) {{
                view.classList.add('active');
            }}
        }}
        
        function renderCharacterChips() {{
            const container = document.getElementById('character-chips');
            
            // 计算每个角色的出场次数（基于全部场景）
            const charCounts = {{}};
            allScenes.forEach(scene => {{
                scene.characters.forEach(c => {{
                    charCounts[c.name] = (charCounts[c.name] || 0) + 1;
                }});
            }});
            
            // 分离已选择和未选择的角色
            const selectedList = Array.from(selectedCharacters).sort((a, b) => (charCounts[b] || 0) - (charCounts[a] || 0));
            const allUnselected = filteredCharacters.filter(c => !selectedCharacters.has(c)).sort((a, b) => (charCounts[b] || 0) - (charCounts[a] || 0));
            const unselectedList = showAll ? allUnselected : allUnselected.slice(0, 20);
            
            let html = '';
            
            // 已选择的角色
            if (selectedList.length > 0) {{
                html += '<p style="width: 100%; margin-bottom: 0.5rem; font-size: 0.875rem; color: #9ca3af;">已选择的角色：</p>';
                html += selectedList.map(char => 
                    `<span class="chip selected" onclick="toggleCharacter('${{char}}')">${{char}} <span style="opacity: 0.7; font-size: 0.75rem;">(${{charCounts[char] || 0}})</span></span>`
                ).join('');
            }}
            
            // 未选择的角色
            if (unselectedList.length > 0) {{
                html += '<p style="width: 100%; margin-bottom: 0.5rem; margin-top: ' + (selectedList.length > 0 ? '1rem' : '0') + '; font-size: 0.875rem; color: #9ca3af;">点击选择更多角色：</p>';
                html += unselectedList.map(char => 
                    `<span class="chip" onclick="toggleCharacter('${{char}}')">${{char}} <span style="opacity: 0.7; font-size: 0.75rem;">(${{charCounts[char] || 0}})</span></span>`
                ).join('');
                
                if (!showAll && allUnselected.length > 20) {{
                    html += `<span class="chip" onclick="expandCharacterList()" style="cursor: pointer; background: rgba(168, 85, 247, 0.2); border-color: #a855f7;">+${{allUnselected.length - 20}} 更多 ↓</span>`;
                }} else if (showAll && allUnselected.length > 20) {{
                    html += `<span class="chip" onclick="collapseCharacterList()" style="cursor: pointer; background: rgba(168, 85, 247, 0.2); border-color: #a855f7;">收起 ↑</span>`;
                }}
            }}
            
            container.innerHTML = html;
            
            document.getElementById('selected-info').textContent = selectedCharacters.size > 0 
                ? `已选择 ${{selectedCharacters.size}} 个角色，共 ${{selectedList.reduce((sum, char) => sum + (charCounts[char] || 0), 0)}} 个场次` 
                : '';
            document.getElementById('clear-btn').style.display = selectedCharacters.size > 0 ? 'block' : 'none';
        }}
        
        function toggleCharacter(char) {{
            if (selectedCharacters.has(char)) {{
                selectedCharacters.delete(char);
            }} else {{
                selectedCharacters.add(char);
            }}
            renderCharacterChips();
            renderTimelineRows();
        }}
        
        function clearSelection() {{
            selectedCharacters.clear();
            renderCharacterChips();
            renderTimelineRows();
        }}
        
        function expandCharacterList() {{
            showAll = true;
            renderCharacterChips();
        }}
        
        function collapseCharacterList() {{
            showAll = false;
            renderCharacterChips();
        }}
        
        function filterCharacters() {{
            const query = document.getElementById('search-input').value.toLowerCase();
            filteredCharacters = characters.filter(c => c.toLowerCase().includes(query));
            showAll = false;
            if (selectedCharacters.size === 0) {{
                renderCharacterChips();
            }}
        }}
        
        function renderTimelineHeader() {{
            const header = document.getElementById('timeline-header');
            if (!header) return;
            // 保留第一个“角色 / 场次”单元，清空后面的
            while (header.children.length > 1) {{
                header.removeChild(header.lastChild);
            }}
            filteredScenes.forEach(scene => {{
                const div = document.createElement('div');
                div.className = 'scene-label';
                div.title = scene.fullTitle;
                div.textContent = scene.sceneNumber;
                header.appendChild(div);
            }});
        }}
        
        function renderTimelineRows() {{
            const container = document.getElementById('timeline-rows');
            
            if (selectedCharacters.size === 0) {{
                container.innerHTML = '<div style="text-align: center; padding: 3rem; color: #9ca3af;">请选择角色以查看时间线</div>';
                return;
            }}
            
            const displayChars = Array.from(selectedCharacters).sort();
            
            container.innerHTML = displayChars.map(char => {{
                const cells = filteredScenes.map((scene, idx) => {{
                    const hasChar = scene.characters.some(c => c.name === char);
                    return `<div class="timeline-cell ${{hasChar ? 'has-character' : ''}}" 
                                 onclick="${{hasChar ? `showCellDetail('${{char}}', ${{idx}})` : ''}}"
                                 title="${{hasChar ? char + ' 在 ' + scene.fullTitle : ''}}">
                                ${{hasChar ? '<div class="dot"></div>' : ''}}
                            </div>`;
                }}).join('');
                
                return `<div class="timeline-row">
                    <div class="character-label" title="${{char}}">${{char}}</div>
                    ${{cells}}
                </div>`;
            }}).join('');
        }}
        
        function showCellDetail(character, sceneIdx) {{
            const scene = filteredScenes[sceneIdx];
            const charData = scene.characters.find(c => c.name === character);
            
            document.getElementById('modal-title').textContent = `${{character}} - ${{scene.fullTitle}}`;
            document.getElementById('modal-body').innerHTML = `
                <div class="scene-info">
                    <div><span>场景类型：</span> ${{scene.sceneCategory}}</div>
                    <div><span>光线：</span> ${{scene.lighting}}</div>
                    <div><span>空间：</span> ${{scene.space}}</div>
                    <div><span>地区：</span> ${{scene.region}}</div>
                    <div style="grid-column: 1 / -1;"><span>地点：</span> ${{scene.mainLocation}}${{scene.subLocation ? ' - ' + scene.subLocation : ''}}</div>
                </div>
                <div class="status-box" style="margin-top: 1rem;">
                    <h4>角色状态</h4>
                    <p>${{charData.status}}</p>
                </div>
                <div class="scene-summary">
                    <h4>剧情概要</h4>
                    <p>${{scene.summary}}</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem;">该场次其他角色</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        ${{scene.characters.filter(c => c.name !== character).map(c => 
                            `<span class="chip">${{c.name}}</span>`
                        ).join('') || '<span style="font-size: 0.875rem; color: #9ca3af;">仅此角色</span>'}}
                    </div>
                </div>
            `;
            document.getElementById('modal').classList.add('show');
        }}
        
        function showCharacterTimeline() {{
            const select = document.getElementById('character-select');
            const idx = parseInt(select.value);
            const container = document.getElementById('character-timeline');
            
            if (isNaN(idx)) {{
                container.innerHTML = '<div class="empty-state"><p style="font-size: 1.125rem;">请选择一个角色查看其时间线</p></div>';
                return;
            }}
            
            const timeline = timelines[idx];
            const scenesForVersion = timeline.scenes.filter(
                s => currentVersion === 'all' || (s.version || '') === currentVersion
            );
            
            if (scenesForVersion.length === 0) {{
                container.innerHTML = '<div class="empty-state"><p style="font-size: 1.125rem;">当前版本下该角色无场次</p></div>';
                return;
            }}

            container.innerHTML = scenesForVersion.map(scene => `
                <div class="timeline-card">
                    <h3>${{scene.fullTitle}}</h3>
                    <div class="status-box">
                        <h4>状态</h4>
                        <p>${{scene.status}}</p>
                    </div>
                    <div class="plot-box">
                        <h4>剧情</h4>
                        <p>${{scene.summary}}</p>
                    </div>
                </div>
            `).join('');
        }}
        
        function closeModal(event) {{
            if (!event || event.target.id === 'modal') {{
                document.getElementById('modal').classList.remove('show');
            }}
        }}
        
        function onVersionChange() {{
            const select = document.getElementById('version-select');
            currentVersion = select ? select.value : 'all';
            
            if (currentVersion === 'all') {{
                filteredScenes = allScenes.slice();
            }} else {{
                filteredScenes = allScenes.filter(
                    s => (s.version || '') === currentVersion
                );
            }}

            const hint = document.getElementById('version-hint');
            if (hint) {{
                hint.textContent = currentVersion === 'all'
                    ? '当前显示：全部版本'
                    : '当前显示：' + currentVersion;
            }}
            
            renderTimelineHeader();
            renderTimelineRows();
            applyVersionFilterToScenes();
            showCharacterTimeline();
        }}

        function applyVersionFilterToScenes() {{
            const cards = document.querySelectorAll('.scene-card');
            cards.forEach(card => {{
                const v = card.getAttribute('data-version') || '';
                if (currentVersion === 'all' || v === currentVersion) {{
                    card.style.display = '';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    import sys
    
    # 默认文件名
    csv_file = 'data/sql/CharacterStatus.csv'
    output_file = 'CharacterStatus.html'
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"📖 正在读取 CSV 文件: {csv_file}")
    scenes, characters, timelines = load_and_process_csv(csv_file)
    
    print(f"✅ 数据加载完成:")
    print(f"   - 场次数: {len(scenes)}")
    print(f"   - 角色数: {len(characters)}")
    
    print(f"🎨 正在生成 HTML 文件: {output_file}")
    generate_html(scenes, characters, timelines, output_file)
    
    print(f"✅ HTML 文件生成成功！")
    print(f"📂 输出文件: {output_file}")


if __name__ == '__main__':
    main()
