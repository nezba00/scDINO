"""All Dash callbacks for the UMAP explorer app."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from ..embedding import compute_embedding
from .figures import build_main_figure

# Max display length for class names in the UI
_MAX_CLASS_LEN = 16

# Panel IDs in tab order
_PANELS = ["panel-view", "panel-tracks", "panel-labels", "panel-umap", "panel-ml"]
_TAB_TO_PANEL = {
    "tab-view": "panel-view",
    "tab-tracks": "panel-tracks",
    "tab-labels": "panel-labels",
    "tab-umap": "panel-umap",
    "tab-ml": "panel-ml",
}


def _trunc(text: str, maxlen: int = _MAX_CLASS_LEN) -> str:
    """Truncate text with ellipsis if too long."""
    s = str(text)
    return s[:maxlen - 2] + ".." if len(s) > maxlen else s


def register(app):
    """Register all callbacks on the Dash app instance."""

    # ------------------------------------------------------------------ #
    #  Tab visibility toggle
    # ------------------------------------------------------------------ #

    @app.callback(
        [Output(pid, "style") for pid in _PANELS],
        Input("sidebar-tabs", "value"),
    )
    def toggle_tabs(tab_value):
        active = _TAB_TO_PANEL.get(tab_value, "panel-view")
        return [
            {"display": "block"} if pid == active else {"display": "none"}
            for pid in _PANELS
        ]

    # ------------------------------------------------------------------ #
    #  Main figure update
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("umap-graph", "figure"),
        Output("status-bar", "children"),
        Input("point-opacity", "value"),
        Input("point-size", "value"),
        Input("color-by", "value"),
        Input("drag-mode", "value"),
        Input("density-toggles", "value"),
        Input("phenotype-filter", "value"),
        Input("track-options", "value"),
        Input("trajectory-color-mode", "value"),
        Input("figure-trigger", "data"),
        Input("umap-store", "data"),
    )
    def update_figure(
        opacity, point_size, color_by, drag_mode,
        density_toggles, phenotype_filter,
        track_options, trajectory_color_mode,
        figure_trigger, umap_store,
    ):
        from .app import state
        if state is None:
            raise PreventUpdate

        opacity = opacity if opacity is not None else 0.7
        point_size = point_size if point_size is not None else 4
        color_by = color_by or "phenotype"
        drag_mode = drag_mode or "zoom"
        density_toggles = density_toggles or []
        phenotype_filter = phenotype_filter or []
        track_options = track_options or []
        trajectory_color_mode = trajectory_color_mode or "Track"

        state.update_filter(
            phenotype_filter=phenotype_filter if phenotype_filter else None,
            track_filter="filter" in track_options,
        )
        state.update_neighbors_model()

        fig = build_main_figure(
            state,
            color_by=color_by,
            opacity=opacity,
            point_size=point_size,
            show_density="density" in density_toggles,
            per_phenotype_density="per_phenotype" in density_toggles,
            show_trajectories="trajectories" in track_options,
            trajectory_color_mode=trajectory_color_mode,
            drag_mode=drag_mode,
        )

        n_filtered = len(state.df_filtered)
        n_total = len(state.df)
        if n_filtered < n_total:
            status = f"{n_filtered:,} / {n_total:,} points"
        else:
            status = f"{n_total:,} points loaded"

        return fig, status

    # ------------------------------------------------------------------ #
    #  Click → show neighbors
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("coords-display", "children"),
        Output("neighbors-list", "children"),
        Input("umap-graph", "clickData"),
    )
    def on_click(click_data):
        from .app import state
        if state is None or click_data is None:
            raise PreventUpdate

        point = click_data["points"][0]
        point_idx = point.get("pointIndex")
        if point_idx is None:
            raise PreventUpdate

        try:
            original_idx = state.df_filtered.index[point_idx]
        except IndexError:
            raise PreventUpdate

        row = state.df.loc[original_idx]
        coords_text = f"({row['umap_1']:.3f}, {row['umap_2']:.3f})"

        if state.nbrs is None:
            return coords_text, "KNN model not available"

        X_filt = state.df_filtered[["umap_1", "umap_2"]].values
        filt_loc = state.df_filtered.index.get_loc(original_idx)
        distances, indices = state.nbrs.kneighbors(
            X_filt[filt_loc].reshape(1, -1)
        )

        neighbor_rows = state.df_filtered.iloc[indices.flatten()].copy()
        neighbor_rows["_dist"] = distances.flatten()

        _, color_mapping, _ = state.color_map(state.df["phenotype"])

        nn_items = []
        for _, nrow in neighbor_rows.iterrows():
            track_id = nrow["track_id"]
            t_val = nrow.get("t", "?")
            pheno = str(nrow.get("phenotype", "?"))
            dist = nrow["_dist"]
            dot_color = color_mapping.get(pheno, "#AAAAAA")
            nn_items.append(
                html.Div(
                    className="nn-row",
                    children=[
                        html.Span(className="color-dot",
                                  style={"backgroundColor": dot_color}),
                        html.Span(f"#{track_id}", className="nn-field"),
                        html.Span(f"t={t_val}", className="nn-field"),
                        html.Span(_trunc(pheno), className="nn-field",
                                  title=pheno),
                        html.Span(f"{dist:.3f}", className="nn-dist"),
                    ],
                )
            )

        return coords_text, nn_items

    # ------------------------------------------------------------------ #
    #  Lasso/box select → stats
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("selection-stats", "children"),
        Input("umap-graph", "selectedData"),
        prevent_initial_call=True,
    )
    def on_select(selected_data):
        from .app import state
        if state is None or selected_data is None:
            raise PreventUpdate

        points = selected_data.get("points", [])
        if not points:
            raise PreventUpdate

        point_indices = [p["pointIndex"] for p in points if "pointIndex" in p]
        if not point_indices:
            raise PreventUpdate

        sel_idx = [state.df_filtered.index[i] for i in point_indices
                   if i < len(state.df_filtered)]

        # Store selection indices on state for apply-label
        state._last_selection = sel_idx

        selected = state.df_filtered.loc[sel_idx]
        counts = selected["phenotype"].value_counts()
        total = counts.sum()
        _, color_mapping, _ = state.color_map(state.df["phenotype"])

        # Color-coded distribution bar
        bar_segments = []
        for pheno, count in counts.items():
            pct = count / total * 100
            dot_color = color_mapping.get(str(pheno), "#AAAAAA")
            bar_segments.append(
                html.Div(
                    style={
                        "width": f"{pct}%",
                        "backgroundColor": dot_color,
                        "height": "100%",
                    },
                    title=f"{pheno}: {pct:.1f}%",
                )
            )
        dist_bar = html.Div(
            className="distribution-bar",
            children=bar_segments,
        )

        # Per-class rows with color dots, truncated names, right-aligned pct
        stats_rows = []
        for pheno, count in counts.items():
            pct = count / total * 100
            dot_color = color_mapping.get(str(pheno), "#AAAAAA")
            stats_rows.append(
                html.Div(
                    className="stats-row",
                    children=[
                        html.Span(className="color-dot",
                                  style={"backgroundColor": dot_color}),
                        html.Span(_trunc(str(pheno)), className="stats-name",
                                  title=str(pheno)),
                        html.Span(f"{count} ({pct:.1f}%)", className="stats-pct"),
                    ],
                )
            )
        stats_content = html.Div([
            html.Div(f"Selected {len(sel_idx)} points", style={"marginBottom": "4px"}),
            dist_bar,
            *stats_rows,
        ])
        return stats_content

    # ------------------------------------------------------------------ #
    #  Label management
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("labels-store", "data"),
        Output("label-list", "children"),
        Output("annotation-text", "value"),
        Output("active-label-store", "data"),
        Output("active-label-display", "children"),
        Output("figure-trigger", "data", allow_duplicate=True),
        Input("add-label-btn", "n_clicks"),
        Input({"type": "label-select-btn", "index": ALL}, "n_clicks"),
        Input({"type": "label-delete-btn", "index": ALL}, "n_clicks"),
        State("annotation-text", "value"),
        State("labels-store", "data"),
        State("active-label-store", "data"),
        State("figure-trigger", "data"),
        prevent_initial_call=True,
    )
    def manage_labels(add_click, select_clicks, delete_clicks,
                      text_value, labels, active_label, trigger):
        """Unified label management: add, select, delete."""
        from .app import state
        if state is None:
            raise PreventUpdate

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        labels = labels or []
        trigger_val = trigger or 0
        fig_trigger = no_update

        prop_id = ctx.triggered[0]["prop_id"]

        # Add new label
        if "add-label-btn" in prop_id:
            new_label = (text_value or "").strip()
            if new_label and new_label not in labels:
                labels.append(new_label)
                active_label = new_label

        # Select a label
        elif "label-select-btn" in prop_id:
            try:
                info = json.loads(prop_id.rsplit(".", 1)[0])
                active_label = info["index"]
            except Exception:
                pass

        # Delete a label
        elif "label-delete-btn" in prop_id:
            try:
                info = json.loads(prop_id.rsplit(".", 1)[0])
                lbl = info["index"]
                if lbl in labels:
                    labels.remove(lbl)
                # Remove from all labeled points
                if state is not None:
                    mask = state.df["label_manual"] == lbl
                    state.df.loc[mask, "label_manual"] = "unlabeled"
                    fig_trigger = trigger_val + 1
                if active_label == lbl:
                    active_label = labels[0] if labels else None
            except Exception:
                pass

        # Build label list UI
        label_ui = _build_label_list(labels, active_label)
        active_display = _active_label_display(active_label)

        return labels, label_ui, "", active_label, active_display, fig_trigger

    # Apply active label to current selection
    @app.callback(
        Output("selection-stats", "children", allow_duplicate=True),
        Output("figure-trigger", "data", allow_duplicate=True),
        Input("apply-label-btn", "n_clicks"),
        State("active-label-store", "data"),
        State("figure-trigger", "data"),
        prevent_initial_call=True,
    )
    def apply_label(n_clicks, active_label, trigger):
        from .app import state
        if state is None or not n_clicks:
            raise PreventUpdate

        if not active_label:
            return html.Div("No label selected", style={"color": "#DC322F"}), no_update

        sel_idx = getattr(state, "_last_selection", None)
        if not sel_idx:
            return html.Div("No points selected", style={"color": "#DC322F"}), no_update

        state.df.loc[sel_idx, "label_manual"] = active_label
        state.df_filtered = state.df.loc[state.df_filtered.index]

        msg = html.Div(
            f"Labeled {len(sel_idx)} points as '{active_label}'",
            style={"color": "#859900"},
        )
        return msg, (trigger or 0) + 1

    def _build_label_list(labels, active_label):
        """Build the scrollable label list with select/delete buttons."""
        if not labels:
            return html.Div("No labels defined", style={"color": "#93A1A1", "fontSize": "11px"})
        items = []
        for lbl in labels:
            is_active = lbl == active_label
            items.append(
                html.Div(
                    className="label-row" + (" label-row-active" if is_active else ""),
                    children=[
                        html.Button(
                            _trunc(lbl, 20),
                            id={"type": "label-select-btn", "index": lbl},
                            className="label-select-btn",
                            title=lbl,
                        ),
                        html.Button(
                            "x",
                            id={"type": "label-delete-btn", "index": lbl},
                            className="label-delete-btn",
                            title=f"Delete '{lbl}'",
                        ),
                    ],
                )
            )
        return html.Div(items)

    def _active_label_display(active_label):
        if not active_label:
            return html.Span("None selected", style={"color": "#93A1A1"})
        return html.Span(active_label, style={"color": "#268BD2", "fontWeight": "700"})

    # ------------------------------------------------------------------ #
    #  Track management
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("track-list", "children"),
        Output("figure-trigger", "data", allow_duplicate=True),
        Output("track-text-input", "value"),
        Input("track-add-btn", "n_clicks"),
        Input("track-text-input", "value"),
        Input("track-clear-btn", "n_clicks"),
        State("track-dropdown", "value"),
        State("figure-trigger", "data"),
        prevent_initial_call=True,
    )
    def manage_tracks(add_clicks, text_input, clear_clicks, dropdown_value, trigger):
        from .app import state
        if state is None:
            raise PreventUpdate

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        trigger_val = trigger or 0

        if trigger_id == "track-clear-btn":
            state.selected_tracks = []
            return "None", trigger_val + 1, ""

        if trigger_id == "track-add-btn" and dropdown_value is not None:
            if dropdown_value not in state.selected_tracks:
                state.selected_tracks.append(dropdown_value)
            return _track_list_text(state), trigger_val + 1, no_update

        if trigger_id == "track-text-input" and text_input:
            try:
                candidate_ids = [int(x.strip()) for x in text_input.split(",") if x.strip().isdigit()]
            except Exception:
                return _track_list_text(state), no_update, ""

            valid_set = set(state.df["track_id"].unique())
            new_tracks = [tid for tid in candidate_ids
                          if tid in valid_set and tid not in state.selected_tracks]
            state.selected_tracks.extend(new_tracks)
            return _track_list_text(state), trigger_val + 1, ""

        raise PreventUpdate


    def _track_list_text(state) -> str:
        if not state.selected_tracks:
            return "None"
        return ", ".join(str(t) for t in state.selected_tracks)

    # ------------------------------------------------------------------ #
    #  Recompute UMAP
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("umap-store", "data"),
        Output("umap-recompute-btn", "children"),
        Input("umap-recompute-btn", "n_clicks"),
        State("umap-n-neighbors", "value"),
        State("umap-min-dist", "value"),
        State("umap-topometry", "value"),
        State("umap-store", "data"),
        prevent_initial_call=True,
    )
    def recompute_umap(n_clicks, n_neighbors, min_dist, topometry_val, store):
        from .app import state
        if state is None or not n_clicks:
            raise PreventUpdate

        topometry = bool(topometry_val and "on" in topometry_val)

        embedding = compute_embedding(
            state.feats,
            topometry=topometry,
            n_neighbors=n_neighbors or 50,
            min_dist=min_dist if min_dist is not None else 0.1,
            random_state=42,
        )
        state.df[["umap_1", "umap_2"]] = embedding
        state.update_neighbors_model()

        return (store or 0) + 1, "Recompute UMAP"

    # ------------------------------------------------------------------ #
    #  Retrain classifier
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("ml-report-output", "children", allow_duplicate=True),
        Output("figure-trigger", "data", allow_duplicate=True),
        Input("ml-retrain-btn", "n_clicks"),
        State("ml-train-source", "value"),
        State("figure-trigger", "data"),
        prevent_initial_call=True,
    )
    def retrain_classifier(n_clicks, train_source, trigger):
        from .app import state
        if state is None or not n_clicks:
            raise PreventUpdate

        label_col = train_source or "phenotype"
        report = state.classifier.train(state.df, state.feats, label_col=label_col)
        return report.summary(), (trigger or 0) + 1

    # ------------------------------------------------------------------ #
    #  Show classifier report
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("ml-report-output", "children"),
        Input("ml-report-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def show_report(n_clicks):
        from .app import state
        if state is None or not n_clicks:
            raise PreventUpdate

        report = state.classifier.last_report
        if report is None:
            return "No classifier trained yet."
        return report.summary()

    # ------------------------------------------------------------------ #
    #  Clear all annotations
    # ------------------------------------------------------------------ #

    @app.callback(
        Output("figure-trigger", "data", allow_duplicate=True),
        Output("labels-store", "data", allow_duplicate=True),
        Output("label-list", "children", allow_duplicate=True),
        Output("active-label-store", "data", allow_duplicate=True),
        Output("active-label-display", "children", allow_duplicate=True),
        Input("clear-labels-btn", "n_clicks"),
        State("figure-trigger", "data"),
        prevent_initial_call=True,
    )
    def clear_labels(n_clicks, trigger):
        from .app import state
        if state is None or not n_clicks:
            raise PreventUpdate

        state.df["label_manual"] = "unlabeled"
        return (
            (trigger or 0) + 1,
            [],
            html.Div("No labels defined", style={"color": "#93A1A1", "fontSize": "11px"}),
            None,
            html.Span("None selected", style={"color": "#93A1A1"}),
        )
