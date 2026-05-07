import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

try:
    from streamlit_agraph import Node, Edge
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

LANG_COLORS = {
    "Horn":          "#8b5cf6",
    "Anti-Horn":     "#a78bfa",
    "ConjUI":        "#06b6d4",
    "Square CNF":    "#06b6d4",  # Legacy alias
    "Square2CNF":    "#f59e0b",
    "Affine (XOR)":  "#eab308",
    "BEST_PER_NODE": "#22c55e",
}

def build_agraph_data(tree_model, trace_x=None):
    nodes_out, edges_out = [], []
    root = tree_model.root_
    if root is None:
        return nodes_out, edges_out

    def _walk(node, parent_id=None, edge_label=""):
        node_id = str(id(node))
        
        is_leaf = node['is_leaf']
        is_active = (node.get('active', False) or node_id == str(id(root))) if trace_x is not None else False
        
        goes_left = False
        if trace_x is not None and is_active and not is_leaf:
            try:
                x_2d = trace_x.reshape(1, -1)
                goes_left = bool(node['predicate'].evaluate(x_2d)[0])
            except (AttributeError, Exception):
                pass
            
        color_l = "#64ffda" if (is_active and trace_x is not None and goes_left) else "#475569"
        color_r = "#64ffda" if (is_active and trace_x is not None and not goes_left) else "#475569"

        if is_leaf:
            lbl = f"Class {node['class']}\n(n={node['samples']})"
            color = "#10b981" if node['class'] == 1 else "#ef4444"
            sz = min(35, max(15, node['samples'] * 2))
            nodes_out.append(Node(
                id=node_id, label=lbl, size=sz, shape="circle",
                color=dict(background=color, border="#0f172a"),
                font=dict(color="white", size=14, bold=True)
            ))
        else:
            lang = node.get('language', '')
            if lang == "HORN":  c, shp = "#8b5cf6", "box"
            elif lang == "ANTI_HORN": c, shp = "#a78bfa", "box"
            elif lang in ("CONJ_UI", "SQUARE_CNF"): c, shp = "#06b6d4", "hexagon"
            elif lang == "SQUARE_2CNF": c, shp = "#f59e0b", "hexagon"
            elif lang == "AFFINE": c, shp = "#eab308", "diamond"
            else: c, shp = "#94a3b8", "box"

            pred_str = str(node['predicate']).replace(" <= ", "≤").replace(" >= ", "≥")
            lbl = f"{pred_str}\n(n={node['samples']})"
            sz = min(30, max(15, node['samples']))
            
            b_color = "#64ffda" if is_active and trace_x is not None else "#1e293b"
            b_width = 3 if is_active and trace_x is not None else 1
            
            nodes_out.append(Node(
                id=node_id, label=lbl, size=sz, shape=shp,
                color=dict(background=c, border=b_color),
                borderWidth=b_width,
                font=dict(color="white", size=12)
            ))

            if 'left' in node:
                _walk(node['left'], node_id, "True")
                if is_active and trace_x is not None and goes_left:
                    node['left']['active'] = True
                edges_out.append(Edge(
                    source=node_id, target=str(id(node['left'])), label="T",
                    color=dict(color=color_l), width=3 if goes_left else 1
                ))
            if 'right' in node:
                _walk(node['right'], node_id, "False")
                if is_active and trace_x is not None and not goes_left:
                    node['right']['active'] = True
                edges_out.append(Edge(
                    source=node_id, target=str(id(node['right'])), label="F",
                    color=dict(color=color_r), width=3 if not goes_left else 1
                ))

    _walk(root)
    return nodes_out, edges_out

def radar_chart(df: pd.DataFrame) -> go.Figure:
    df_norm = df.copy()
    df_norm["Test Acc"] = (df["Test Acc"] - df["Test Acc"].min()) / (df["Test Acc"].max() - df["Test Acc"].min() + 1e-9)
    df_norm["Nodes"] = 1 - (df["Nodes"] - df["Nodes"].min()) / (df["Nodes"].max() - df["Nodes"].min() + 1e-9)
    df_norm["Avg |AXp|"] = 1 - (df["Avg |AXp|"] - df["Avg |AXp|"].min()) / (df["Avg |AXp|"].max() - df["Avg |AXp|"].min() + 1e-9)
    fig = go.Figure()
    for _, row in df_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["Test Acc"], row["Nodes"], row["Avg |AXp|"], row["Test Acc"]],
            theta=["Accuracy", "Size Efficiency", "AXp Conciseness", "Accuracy"],
            fill='toself', name=row["Language"], line_color=LANG_COLORS.get(row["Language"])
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, template="plotly_dark"
    )
    return fig

def axp_feature_heatmap(axp_data, n_feats, lang_name) -> go.Figure:
    if not axp_data: return None
    freqs = np.zeros((1, n_feats))
    for d in axp_data:
        for f in d["axp_features"]:
            if f < n_feats:
                freqs[0, f] += 1
    freqs = freqs / len(axp_data)
    active_cols = np.where(freqs[0] > 0)[0]
    if len(active_cols) == 0: return None
    
    fig = px.imshow(
        freqs[:, active_cols], x=[f"f{i}" for i in active_cols], y=[lang_name],
        color_continuous_scale="Teal", title="Feature participation frequency in AXp",
        labels=dict(x="Feature Index", color="Frequency"), zmin=0, zmax=1
    )
    fig.update_layout(template="plotly_dark", font=dict(family="Inter"), height=250)
    return fig

def plot_calibration_curves(results: dict, X_te, y_te):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#475569", width=1), name="Perfect calibration"
    ))
    for lang_name, res in results.items():
        tree = res["tree"]
        try:
            proba = tree.predict_proba(X_te)[:, 1]
        except AttributeError:
            proba = np.array([float(tree.predict_leaf_proba(X_te[i:i+1])[0]) for i in range(len(X_te))])
        try:
            fraction_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="uniform")
            ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
            fig.add_trace(go.Scatter(
                x=mean_pred, y=fraction_pos, mode="lines+markers", name=f"{lang_name} (ECE={ece:.3f})",
                line=dict(color=LANG_COLORS.get(lang_name, "#888"), width=2), marker=dict(size=8)
            ))
        except Exception:
            pass
    fig.update_layout(
        template="plotly_dark", title="Calibration curves (reliability diagrams)",
        xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), font=dict(family="Inter")
    )
    return fig

def axp_cross_language_heatmap(results: dict, n_feats: int) -> go.Figure:
    langs_with_axp = {lang: res for lang, res in results.items() if res.get("axp_data")}
    if not langs_with_axp: return None
    freq_matrix = np.zeros((len(langs_with_axp), n_feats))
    lang_labels = list(langs_with_axp.keys())
    for i, (lang_name, res) in enumerate(langs_with_axp.items()):
        for d in res["axp_data"]:
            for feat in d["axp_features"]:
                if feat < n_feats: freq_matrix[i, feat] += 1
        if len(res["axp_data"]) > 0: freq_matrix[i] /= len(res["axp_data"])
    active_feats = np.where(freq_matrix.sum(axis=0) > 0)[0]
    if len(active_feats) == 0: return None
    fig = px.imshow(
        freq_matrix[:, active_feats], y=lang_labels, x=[f"f{i}" for i in active_feats], color_continuous_scale="Teal",
        title="Cross-language AXp feature participation rate", labels=dict(x="Feature", y="Language", color="P(∈ AXp)"),
        zmin=0, zmax=1
    )
    fig.update_layout(template="plotly_dark", font=dict(family="JetBrains Mono"), xaxis=dict(tickangle=45))
    return fig
