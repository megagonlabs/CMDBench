from typing import Dict, Tuple, Any, List
import random
import graphviz


def _split_to_multilines(s, maxlen=20):
    tokens = s.split()
    res = []
    curr_line_tokens = []
    curr_line_len = -1
    for t in tokens:
        if curr_line_len >= maxlen:
            res.append(' '.join(curr_line_tokens))
            curr_line_tokens = []
            curr_line_len = -1
        curr_line_tokens.append(t)
        curr_line_len += len(t) + 1
    if len(curr_line_tokens):
        res.append(' '.join(curr_line_tokens))
    return '\n'.join(res)


def graph2graphviz(node2name: Dict[str, List[str]], edge2label: Dict[Tuple[str, str], str], id2color, dpi=64,
                   transparent=True):
    random.seed(0)

    color_profile = {
        'orange': {
            'fill': '#ffcc99',
            'edge': '#ff6600',
        },
        'blue': {
            'fill': '#99ccff',
            'edge': '#0066ff',
        },
        'green': {
            'fill': '#99ff99',
            'edge': '#009900',
        },
        'red': {
            'fill': '#ff9999',
            'edge': '#cc0000',
        },
        'yellow': {
            'fill': '#ffff99',
            'edge': '#ffcc00',
        },
        'purple': {
            'fill': '#cc99ff',
            'edge': '#6600cc',
        },
        'gray': {
            'fill': '#eeeeee',
            'edge': '#666666',
        },
    }

    g = graphviz.Digraph('G')
    g.attr(label='', dpi=str(dpi),
           rankdir='LR', outputorder='edgesfirst', splines='splines',
           compound='true', fontname='Sans Not-Rotated', fontsize='16',
           labelloc='t', labeljust='l', newrank='true',
           bgcolor='transparent' if transparent else '')

    edge_attrs = dict(arrowsize='0.5', penwidth='1.5', arrowhead='dot',
                      color=color_profile['gray']['edge'], style='dashed',
                      fontname='Sans Not-Rotated', fontsize='8', fontcolor=color_profile['gray']['edge'],
                      tailport='_', headport='_')
    for (sv, tv), edge_type in edge2label.items():
        g.edge(str(sv), str(tv), label=edge_type, **edge_attrs)

    node_attrs = dict(shape='rect', height='0.3', margin='0.22,0.055',
                      fontsize='8', fontname='Sans Not-Rotated', style='rounded,filled,bold')
    for node_id in node2name:
        label = node2name[node_id]
        label = _split_to_multilines(label, maxlen=30)
        color = id2color.get(node_id, 'gray')
        color_attrs = dict(
            fillcolor=color_profile[color]['fill'],
            fontcolor=color_profile[color]['edge'],
            color=color_profile[color]['edge']
        )
        g.node(str(node_id), label=label, **node_attrs, **color_attrs)

    # st.write(g.source)

    return g
