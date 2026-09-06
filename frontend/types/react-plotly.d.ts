declare module "react-plotly.js" {
  import * as React from "react";

  export interface PlotlyProps {
    data: unknown;
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (graphDiv: unknown) => void;
    onUpdate?: (graphDiv: unknown) => void;
  }

  export default class Plot extends React.Component<PlotlyProps> {}
}
