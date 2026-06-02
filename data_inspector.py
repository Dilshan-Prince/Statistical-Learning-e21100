"""
data_inspector.py
=================
A modular data sanitization and exploration engine for Google Colab.
"""

from __future__ import annotations

import io
import warnings
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.stats as ss

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
)

# Strings that should be treated as missing values
DEFAULT_GARBAGE_TOKENS = [
    "?", "n/a", "na", "n.a.", "null", "none", "nan", "-", "--",
    "", " ", "missing", "unknown", "<na>"
]


class DataInspector:
    """
    Inspect, clean, normalize and visualize a tabular dataset.
    Methods mutate self.df in place and return self for chaining.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        garbage_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self.df: Optional[pd.DataFrame] = df.copy() if df is not None else None
        self.garbage_tokens = list(garbage_tokens) if garbage_tokens else list(DEFAULT_GARBAGE_TOKENS)

    # ------------------------------------------------------------------ #
    # Internal Helpers
    # ------------------------------------------------------------------ #
    def _has_data(self) -> bool:
        if self.df is None or self.df.empty:
            print("[DataInspector] No data loaded or dataset is empty.")
            return False
        return True

    @property
    def numeric_columns(self) -> list[str]:
        return self.df.select_dtypes(include=np.number).columns.tolist() if self.df is not None else []

    @property
    def categorical_columns(self) -> list[str]:
        return self.df.select_dtypes(exclude=np.number).columns.tolist() if self.df is not None else []

    @staticmethod
    def _parse_csv_list(value: Union[str, Iterable, None]) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    # ------------------------------------------------------------------ #
    # 1. Ingestion & Sanitization
    # ------------------------------------------------------------------ #
    def upload_data(self, filepath: Optional[str] = None, **kwargs) -> "DataInspector":
        """Ingest CSV via Colab widget or local path."""
        if filepath is not None:
            self.df = pd.read_csv(filepath, **kwargs)
        else:
            try:
                from google.colab import files
                uploaded = files.upload()
                if not uploaded:
                    return self
                name = next(iter(uploaded))
                self.df = pd.read_csv(io.BytesIO(uploaded[name]), **kwargs)
                print(f"[DataInspector] Loaded '{name}'.")
            except ImportError:
                raise RuntimeError("Colab upload only available in Google Colab. Use filepath='...' locally.")
        return self

    def sanitize(self) -> "DataInspector":
        """Convert recognized garbage strings to NaN."""
        if not self._has_data(): return self
        
        tokens = {t.strip().lower() for t in self.garbage_tokens}
        
        def _clean(val):
            if isinstance(val, str) and val.strip().lower() in tokens:
                return np.nan
            return val

        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            self.df[col] = self.df[col].map(_clean)
        return self

    def auto_correct_types(self) -> "DataInspector":
        """Force-convert to numeric if it doesn't result in an entirely null column."""
        if not self._has_data(): return self

        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")
            # If the conversion did not destroy all data, keep it
            if not converted.isna().all():
                self.df[col] = converted
        return self

    # ------------------------------------------------------------------ #
    # 2. Structural Analysis & Cleaning
    # ------------------------------------------------------------------ #
    def data_summary(self, preview_rows: int = 20) -> Optional[pd.DataFrame]:
        """Display rows/cols, data types, missing values, and a preview."""
        if not self._has_data(): return None

        print("=" * 50 + "\nDATASET SUMMARY\n" + "=" * 50)
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")
        print(f"Numeric Columns: {len(self.numeric_columns)}")
        print(f"Categorical Columns: {len(self.categorical_columns)}")
        print("-" * 50)
        
        missing = self.df.isna().sum()
        if missing.sum() == 0:
            print("Missing values: None")
        else:
            print("Missing Values:")
            print(missing[missing > 0].to_string())
            
        print("-" * 50)
        print(f"Exact Duplicates: {self.df.duplicated().sum()}")
        print("=" * 50)
        
        preview = self.df.head(preview_rows)
        try:
            from IPython.display import display
            display(preview)
        except ImportError:
            print(preview)
        return preview

    def handle_missing_values(self, strategy: str = "mean", columns: Union[str, Sequence[str], None] = None, constant_value=None) -> "DataInspector":
        """Impute nulls via mean, median, mode, or constant."""
        if not self._has_data(): return self
        
        cols = self._parse_csv_list(columns) or self.df.columns
        cols = [c for c in cols if c in self.df.columns]

        for col in cols:
            if self.df[col].isna().sum() == 0:
                continue
                
            if strategy in ["mean", "median"] and col in self.numeric_columns:
                fill_val = self.df[col].mean() if strategy == "mean" else self.df[col].median()
            elif strategy == "mode":
                fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan
            elif strategy == "constant" and constant_value is not None:
                fill_val = constant_value
            else:
                continue
                
            self.df[col] = self.df[col].fillna(fill_val)
        return self

    def remove_duplicates(self) -> "DataInspector":
        """Prune exact row matches."""
        if self._has_data():
            self.df = self.df.drop_duplicates().reset_index(drop=True)
        return self

    def handle_outliers(self, columns: Union[str, Sequence[str], None] = None, action: str = "flag") -> "DataInspector":
        """IQR-based outlier management (flag or delete)."""
        if not self._has_data(): return self
        
        cols = self._parse_csv_list(columns) or self.numeric_columns
        cols = [c for c in cols if c in self.numeric_columns]
        
        outlier_mask = pd.Series(False, index=self.df.index)
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_mask |= (self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))
            
        if action == "flag":
            self.df["is_outlier"] = outlier_mask
        elif action == "delete":
            self.df = self.df[~outlier_mask].reset_index(drop=True)
        return self

    def delete_rows(self, indices: Union[str, Sequence[int]]) -> "DataInspector":
        """Targeted deletion of rows via index list/string."""
        if not self._has_data(): return self
        idx_list = [int(i) for i in self._parse_csv_list(indices) if int(i) in self.df.index]
        self.df = self.df.drop(index=idx_list).reset_index(drop=True)
        return self

    def delete_columns(self, columns: Union[str, Sequence[str]]) -> "DataInspector":
        """Targeted deletion of columns via list/string."""
        if not self._has_data(): return self
        cols = [c for c in self._parse_csv_list(columns) if c in self.df.columns]
        self.df = self.df.drop(columns=cols)
        return self

    # ------------------------------------------------------------------ #
    # 3. Feature Engineering Preparation
    # ------------------------------------------------------------------ #
    def extract_normalized_numeric_data(self, method: str = "minmax", columns: Union[str, Sequence[str], None] = None) -> pd.DataFrame:
        """Scale numeric features (minmax, standard, robust)."""
        if not self._has_data(): return pd.DataFrame()
        
        cols = self._parse_csv_list(columns) or self.numeric_columns
        cols = [c for c in cols if c in self.numeric_columns]
        if not cols: return pd.DataFrame()

        scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}
        subset = self.df[cols].fillna(self.df[cols].median()) # Scalers reject NaNs
        
        scaled_data = scalers[method].fit_transform(subset)
        return pd.DataFrame(scaled_data, columns=cols, index=self.df.index)

    def extract_normalized_categorical_data(self, method: str = "onehot", columns: Union[str, Sequence[str], None] = None) -> pd.DataFrame:
        """Encode categorical features (onehot, ordinal, uniform)."""
        if not self._has_data(): return pd.DataFrame()
        
        cols = self._parse_csv_list(columns) or self.categorical_columns
        cols = [c for c in cols if c in self.categorical_columns]
        if not cols: return pd.DataFrame()

        subset = self.df[cols].fillna("Missing")

        if method == "onehot":
            return pd.get_dummies(subset, columns=cols, dtype=int)
            
        encoder = OrdinalEncoder()
        encoded = pd.DataFrame(encoder.fit_transform(subset), columns=cols, index=self.df.index)
        
        if method == "uniform":
            for col in cols:
                max_val = max(encoded[col].max(), 1)
                encoded[col] = encoded[col] / max_val
        return encoded

    def merge_processed_data(self, numeric_method: str = "standard", categorical_method: str = "onehot") -> pd.DataFrame:
        """Return a single DataFrame of scaled numerics and encoded categoricals."""
        num_df = self.extract_normalized_numeric_data(method=numeric_method)
        cat_df = self.extract_normalized_categorical_data(method=categorical_method)
        return pd.concat([num_df, cat_df], axis=1)

    # ------------------------------------------------------------------ #
    # 4. Advanced Interactive Visualization
    # ------------------------------------------------------------------ #
    def plot_univariate(self, column: str):
        """Generate a 3-panel subplot: Violin/Box, Scatter, Histogram."""
        if not self._has_data() or column not in self.numeric_columns: return None
        
        series = self.df[column].dropna()
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Distribution", "Index vs Value", "Histogram"))
        
        fig.add_trace(go.Violin(x=series, box_visible=True, meanline_visible=True, orientation="h", name=column), row=1, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="markers", marker=dict(opacity=0.6), name="Values"), row=1, col=2)
        fig.add_trace(go.Histogram(x=series, name="Frequency"), row=1, col=3)
        
        fig.update_layout(title_text=f"Univariate Analysis: {column}", showlegend=False, template="plotly_white")
        return fig

    def plot_relationship(self, x: str, y: str):
        """Auto-detect types and plot Scatter, Box, or Grouped Bar."""
        if not self._has_data() or x not in self.df.columns or y not in self.df.columns: return None
        
        data = self.df[[x, y]].dropna()
        x_num, y_num = x in self.numeric_columns, y in self.numeric_columns
        
        if x_num and y_num:
            # Fallback if statsmodels isn't installed
            try:
                import statsmodels.api  # noqa
                trendline = "ols"
            except ImportError:
                trendline = None
            fig = px.scatter(data, x=x, y=y, trendline=trendline, template="plotly_white", title=f"{y} vs {x}")
        elif x_num != y_num:
            cat, num = (x, y) if not x_num else (y, x)
            fig = px.box(data, x=cat, y=num, points="all", template="plotly_white", title=f"{num} by {cat}")
        else:
            counts = data.groupby([x, y]).size().reset_index(name="count")
            fig = px.bar(counts, x=x, y="count", color=y, barmode="group", template="plotly_white", title=f"{x} vs {y}")
            
        return fig

    def plot_categorical_frequency(self, column: str):
        """Create bar charts displaying raw counts and percentage labels."""
        if not self._has_data() or column not in self.df.columns: return None
        
        counts = self.df[column].value_counts()
        pcts = (counts / counts.sum() * 100).round(1)
        labels = [f"{c} ({p}%)" for c, p in zip(counts.values, pcts.values)]
        
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, text=labels, textposition="outside"))
        fig.update_layout(title=f"Frequency of {column}", template="plotly_white")
        return fig

    # ------------------------------------------------------------------ #
    # 5. Deep Statistical Insights
    # ------------------------------------------------------------------ #
    def plot_all_associations_heatmap(self):
        """Visualize Pearson, Cramér's V, and Eta across all data types."""
        if not self._has_data(): return None
        
        cols = self.df.columns.tolist()
        num_set = set(self.numeric_columns)
        matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                a_num, b_num = a in num_set, b in num_set
                
                if a_num and b_num:
                    val = abs(self.df[[a, b]].corr().iloc[0, 1])
                elif not a_num and not b_num:
                    val = self._cramers_v(self.df[a], self.df[b])
                else:
                    cat, num = (a, b) if not a_num else (b, a)
                    val = self._eta(self.df[cat], self.df[num])
                    
                matrix.loc[a, b] = matrix.loc[b, a] = val if pd.notna(val) else 0

        fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=0, zmax=1, title="Unified Associations Heatmap")
        fig.update_layout(template="plotly_white")
        return fig

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Cramér's V for Categorical-Categorical."""
        confusion = pd.crosstab(x, y)
        if confusion.empty: return np.nan
        chi2 = ss.chi2_contingency(confusion)[0]
        n = confusion.sum().sum()
        r, k = confusion.shape
        return np.sqrt(chi2 / (n * min(k - 1, r - 1))) if n > 0 and min(k - 1, r - 1) > 0 else np.nan

    @staticmethod
    def _eta(cat: pd.Series, num: pd.Series) -> float:
        """Correlation Ratio (Eta via ANOVA) for Categorical-Numeric."""
        df = pd.DataFrame({"cat": cat, "num": num}).dropna()
        if df.empty or df["cat"].nunique() < 2: return np.nan
        
        grand_mean = df["num"].mean()
        grouped = df.groupby("cat")["num"]
        ss_between = (grouped.count() * (grouped.mean() - grand_mean) ** 2).sum()
        ss_total = ((df["num"] - grand_mean) ** 2).sum()
        
        return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0


# ------------------------------------------------------------------ #
# 6. Custom Modular Plotting
# ------------------------------------------------------------------ #
class PlottingMethods:
    """Independent HTML-wrapped chart generators for flexible embedding."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _get_series(self, column: str) -> pd.Series:
        return self.df[column].dropna() if column in self.df.columns else pd.Series(dtype=float)

    def bar_chart(self, column: str, title: str = "Bar Chart") -> str:
        """HTML Bar Chart."""
        counts = self._get_series(column).value_counts()
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color="#4C78A8"))
        fig.update_layout(title=title, template="plotly_white")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def pie_chart(self, column: str, title: str = "Pie Chart") -> str:
        """HTML Pie Chart."""
        counts = self._get_series(column).value_counts()
        fig = go.Figure(go.Pie(labels=counts.index.astype(str), values=counts.values, hole=0.3))
        fig.update_layout(title=title, template="plotly_white")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def histogram(self, column: str, bins: int = 30, title: str = "Histogram") -> str:
        """HTML Histogram."""
        series = pd.to_numeric(self._get_series(column), errors="coerce").dropna()
        fig = go.Figure(go.Histogram(x=series, nbinsx=bins, marker_color="#54A24B"))
        fig.update_layout(title=title, template="plotly_white")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")