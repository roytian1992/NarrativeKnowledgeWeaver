from typing import Dict, List

def generate_html(scenes, chains, output_file):
    """
    生成剧本接戏关系可视化 HTML（严格版）。

    参数：
        scenes: dict，形如
            {
                "scene_1": {
                    "scene_id": "scene_1",
                    "scene_title": "...",
                    "summary": "...",
                    "cmp_info": "..."
                },
                ...
            }
        chains: List[List[str]]，每个元素是一条接戏链里场景 id 的列表
            例如：[["scene_1", "scene_3"], ["scene_2", "scene_5", "scene_6"], ...]
        output_file: str，输出 HTML 文件路径
    """
    print(f"\n正在生成HTML文件: {output_file}")

    # 按链的大小排序
    chains_sorted = sorted(chains, key=len, reverse=True)

    # 颜色方案
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#E74C3C', '#3498DB', '#2ECC71',
        '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
        '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#D35400'
    ]

    # HTML 头部
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>剧本场景接戏关系可视化（严格版）</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
        
        .algorithm-info {
            text-align: center;
            background: #e8f4f8;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }
        
        .algorithm-info strong {
            color: #2980b9;
        }
        
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .filter-section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .filter-label {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .filter-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }
        
        .filter-btn:hover, .filter-btn.active {
            background: #667eea;
            color: white;
        }
        
        .chain-container {
            margin-bottom: 40px;
            display: block;
        }
        
        .chain-container.hidden {
            display: none;
        }
        
        .chain-header {
            background: linear-gradient(90deg, var(--chain-color) 0%, transparent 100%);
            padding: 15px 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .chain-header:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .chain-title {
            font-size: 1.5em;
            font-weight: bold;
            color: white;
        }
        
        .chain-info {
            color: white;
            opacity: 0.9;
            margin-top: 5px;
        }
        
        .chain-badge {
            display: inline-block;
            background: rgba(255,255,255,0.3);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        
        .scenes-flow {
            display: flex;
            align-items: center;
            overflow-x: auto;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .scene-card {
            min-width: 320px;
            max-width: 320px;
            background: white;
            border: 3px solid var(--chain-color);
            border-radius: 15px;
            padding: 20px;
            margin: 0 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        
        .scene-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .scene-arrow {
            font-size: 2em;
            color: var(--chain-color);
            margin: 0 10px;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .scene-title {
            font-size: 1.05em;
            font-weight: bold;
            color: var(--chain-color);
            margin-bottom: 10px;
            border-bottom: 2px solid var(--chain-color);
            padding-bottom: 8px;
            word-wrap: break-word;
        }
        
        .scene-id {
            font-size: 0.85em;
            color: #95a5a6;
            margin-bottom: 10px;
            font-style: italic;
        }
        
        .scene-summary {
            color: #2c3e50;
            line-height: 1.6;
            margin-bottom: 15px;
            font-size: 0.9em;
            max-height: 100px;
            overflow-y: auto;
        }
        
        .scene-cmp {
            background: #ecf0f1;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.8em;
            color: #7f8c8d;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .scene-cmp-title {
            font-weight: bold;
            color: #34495e;
            margin-bottom: 5px;
        }
        
        .toggle-details {
            background: var(--chain-color);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 0.85em;
            transition: all 0.3s;
        }
        
        .toggle-details:hover {
            opacity: 0.8;
            transform: scale(1.05);
        }
        
        .details-content {
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }
        
        .details-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .scenes-flow {
                flex-direction: column;
            }
            
            .scene-arrow {
                transform: rotate(90deg);
                margin: 10px 0;
            }
            
            .scene-card {
                min-width: 100%;
                max-width: 100%;
                margin: 10px 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 剧本场景接戏关系可视化</h1>
        <div class="subtitle">Scene Continuity Relationship Visualization - 严格版</div>
        
        <div class="algorithm-info">
            <strong>算法规则:</strong> 使用最大团（Maximum Clique）算法 - 链中任意两个场景都必须能直接接戏
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">''' + str(len(scenes)) + '''</div>
                <div class="stat-label">总场景数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">''' + str(len(chains)) + '''</div>
                <div class="stat-label">接戏链数量</div>
            </div>
'''

    if chains:
        html_content += '''            <div class="stat-card">
                <div class="stat-number">''' + str(max(len(chain) for chain in chains)) + '''</div>
                <div class="stat-label">最大链场景数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">''' + f"{sum(len(chain) for chain in chains) / len(chains):.1f}" + '''</div>
                <div class="stat-label">平均每链场景数</div>
            </div>
'''

    html_content += '''        </div>
        
        <div class="filter-section">
            <div class="filter-label">筛选接戏链（按场景数）:</div>
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterChains('all', this)">全部显示</button>
                <button class="filter-btn" onclick="filterChains('large', this)">大型链 (≥5场)</button>
                <button class="filter-btn" onclick="filterChains('medium', this)">中型链 (3-4场)</button>
                <button class="filter-btn" onclick="filterChains('small', this)">小型链 (2场)</button>
            </div>
        </div>
'''

    # 每条链的 HTML
    for chain_idx, chain in enumerate(chains_sorted):
        color = colors[chain_idx % len(colors)]
        chain_size = len(chain)

        # 确定链的大小类别
        if chain_size >= 5:
            size_class = 'large'
        elif chain_size >= 3:
            size_class = 'medium'
        else:
            size_class = 'small'

        html_content += f'''
        <div class="chain-container" data-size="{size_class}" style="--chain-color: {color};">
            <div class="chain-header" onclick="toggleChain({chain_idx})">
                <div class="chain-title">
                    接戏链 {chain_idx + 1}
                    <span class="chain-badge">完全连通</span>
                </div>
                <div class="chain-info">包含 {len(chain)} 个场景 | 任意两场景都可直接接戏</div>
            </div>
            
            <div id="chain-{chain_idx}" class="scenes-flow">
    '''

        for i, scene_id in enumerate(chain):
            scene = scenes[scene_id]

            # 转义 & 截断
            summary_raw = scene.get('summary', '') or ''
            cmp_raw = scene.get('cmp_info', '') or ''

            summary_text = summary_raw[:150].replace('<', '&lt;').replace('>', '&gt;')
            if len(summary_raw) > 150:
                summary_text += '...'

            cmp_text = (
                cmp_raw[:500]
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('\n', '<br>')
            )
            if len(cmp_raw) > 500:
                cmp_text += '...'

            html_content += f'''
                <div class="scene-card">
                    <div class="scene-title">{scene.get('scene_title', '')}</div>
                    <div class="scene-id">{scene_id}</div>
                    <div class="scene-summary"><strong>剧情:</strong> {summary_text}</div>
                    <button class="toggle-details" onclick="toggleDetails('{scene_id}_{chain_idx}')">查看服化道详情</button>
                    <div id="details-{scene_id}_{chain_idx}" class="details-content">
                        <div class="scene-cmp">
                            <div class="scene-cmp-title">服化道信息:</div>
                            {cmp_text}
                        </div>
                    </div>
                </div>
            '''

            if i < len(chain) - 1:
                html_content += '''
                <div class="scene-arrow">↔</div>
            '''

        html_content += '''
            </div>
        </div>
    '''

    # 尾部 JS
    html_content += '''
    </div>
    
    <script>
        function toggleDetails(sceneId) {
            const details = document.getElementById('details-' + sceneId);
            if (!details) return;
            details.classList.toggle('active');
        }
        
        function toggleChain(chainIdx) {
            const chain = document.getElementById('chain-' + chainIdx);
            if (!chain) return;
            if (chain.style.display === 'none') {
                chain.style.display = 'flex';
            } else {
                chain.style.display = 'none';
            }
        }
        
        function filterChains(type, btn) {
            // 更新按钮状态
            document.querySelectorAll('.filter-btn').forEach(function(b) {
                b.classList.remove('active');
            });
            if (btn) {
                btn.classList.add('active');
            }
            
            // 筛选链
            const chains = document.querySelectorAll('.chain-container');
            chains.forEach(function(chain) {
                if (type === 'all') {
                    chain.classList.remove('hidden');
                } else {
                    const size = chain.getAttribute('data-size');
                    if (size === type) {
                        chain.classList.remove('hidden');
                    } else {
                        chain.classList.add('hidden');
                    }
                }
            });
        }
    </script>
</body>
</html>
'''

    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML 文件生成成功: {output_file}")
    print(f"包含 {len(chains_sorted)} 条接戏链（严格版）")
