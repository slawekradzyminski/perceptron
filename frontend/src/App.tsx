import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { AfterUpdatePanel } from "./components/AfterUpdatePanel";
import { BoundaryPanel } from "./components/BoundaryPanel";
import { CustomModal } from "./components/CustomModal";
import { ErrorSurfacePanel } from "./components/ErrorSurfacePanel";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { Header } from "./components/Header";
import { LmsPage } from "./components/LmsPage";
import { MlpInternalsPanel } from "./components/MlpInternalsPanel";
import { StepMathPanel } from "./components/StepMathPanel";
import { SwitchboardPanel } from "./components/SwitchboardPanel";
import { useDiagnosticsApi } from "./hooks/useDiagnosticsApi";
import { usePerceptronApi } from "./hooks/usePerceptronApi";
import { useHotkeys } from "./hooks/useHotkeys";
import type { TooltipState } from "./types";
import { defaultCustomConfig, pointsForDataset } from "./utils/custom";
import {
  contributionGrid,
  createFilledGrid,
  makeGrid,
  resizeGrid,
  reshapeToGrid,
} from "./utils/visuals";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

type AppRoute = "main" | "diagnostics" | "lms";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [lr, setLr] = useState(1);
  const [customConfig, setCustomConfig] = useState(defaultCustomConfig());
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const route: AppRoute =
    location.pathname === "/diagnostics" ? "diagnostics" : location.pathname === "/lms" ? "lms" : "main";
  const [errorSurfaceConfig, setErrorSurfaceConfig] = useState({
    steps: 25,
    wMin: -2,
    wMax: 2,
    bias: 0,
  });
  const [mlpHiddenDim, setMlpHiddenDim] = useState(2);
  const [mlpSampleIndex, setMlpSampleIndex] = useState(0);
  const [mlpLr, setMlpLr] = useState(0.5);
  const [mlpSeed, setMlpSeed] = useState(0);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    text: "",
    x: 0,
    y: 0,
  });

  const { state, reset, step, setState } = usePerceptronApi({
    datasetName: "or",
    gridRows: 1,
    gridCols: 2,
    sampleCount: 4,
    sampleIdx: 0,
    w: [0, 0],
    b: 0,
    lastStep: null,
    nextInput: null,
    apiError: null,
    customApplied: false,
  });

  const plotRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLCanvasElement | null>(null);
  const weightsBeforeRef = useRef<HTMLCanvasElement | null>(null);
  const weightsAfterRef = useRef<HTMLCanvasElement | null>(null);
  const contribRef = useRef<HTMLCanvasElement | null>(null);

  const dim = state.gridRows * state.gridCols;
  const points = useMemo(
    () => pointsForDataset(state.datasetName, dim, customConfig),
    [state.datasetName, dim, customConfig],
  );
  const showBoundary = dim === 2;
  const canFetchDiagnostics = state.datasetName !== "custom" || state.customApplied;
  const showTrainingControls = route === "main";
  const fallbackInput = useMemo(() => Array.from({ length: dim }, () => -1), [dim]);

  const sample = state.lastStep?.x ?? state.nextInput?.x ?? fallbackInput;
  const inputGrid = useMemo(
    () => reshapeToGrid(sample, state.gridRows, state.gridCols),
    [sample, state.gridRows, state.gridCols],
  );
  const weightsAfter = useMemo(
    () => reshapeToGrid(state.w, state.gridRows, state.gridCols),
    [state.w, state.gridRows, state.gridCols],
  );
  const weightsBefore = useMemo(() => {
    if (!state.lastStep) return weightsAfter;
    return reshapeToGrid(
      state.w.map((val, i) => val - state.lastStep!.deltaW[i]),
      state.gridRows,
      state.gridCols,
    );
  }, [state.lastStep, state.w, state.gridRows, state.gridCols, weightsAfter]);
  const biasBefore = state.lastStep ? state.b - state.lastStep.deltaB : state.b;
  const contribGrid = useMemo(
    () => contributionGrid(inputGrid, weightsBefore),
    [inputGrid, weightsBefore],
  );
  const displayScore = state.lastStep
    ? state.lastStep.score
    : contribGrid.flat().reduce((acc, v) => acc + v, 0) + biasBefore;

  const apiConfig = useMemo(() => ({
    apiBase,
    lr,
    customConfig,
  }), [apiBase, lr, customConfig]);

  const diagnostics = useDiagnosticsApi(apiBase, state.datasetName, customConfig);
  const mlpConfig = useMemo(
    () => ({
      hiddenDim: mlpHiddenDim,
      sampleIndex: mlpSampleIndex,
      lr: mlpLr,
      seed: mlpSeed,
    }),
    [mlpHiddenDim, mlpSampleIndex, mlpLr, mlpSeed],
  );

  const handleDatasetChange = (value: string) => {
    setState((prev) => ({ ...prev, datasetName: value, apiError: null }));
    if (value === "custom") {
      setState((prev) => ({
        ...prev,
        gridRows: customConfig.rows,
        gridCols: customConfig.cols,
        customApplied: false,
      }));
      setCustomModalOpen(true);
      return;
    }
    setCustomModalOpen(false);
    void reset(apiConfig, value);
  };

  const handleStep = () => {
    if (!showTrainingControls) return;
    void step(apiConfig);
  };
  const handleReset = () => {
    if (!showTrainingControls) return;
    void reset(apiConfig, state.datasetName);
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: showTrainingControls });

  const updateCustomRows = (rows: number) => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      rows,
      samples: prev.samples.map((sample) => ({
        ...sample,
        grid: resizeGrid(sample.grid, rows, prev.cols, -1),
      })),
    }));
    if (state.datasetName === "custom") {
      setState((prev) => ({ ...prev, gridRows: rows }));
    }
  };

  const updateCustomCols = (cols: number) => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      cols,
      samples: prev.samples.map((sample) => ({
        ...sample,
        grid: resizeGrid(sample.grid, prev.rows, cols, -1),
      })),
    }));
    if (state.datasetName === "custom") {
      setState((prev) => ({ ...prev, gridCols: cols }));
    }
  };

  const addCustomSample = () => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      samples: [
        ...prev.samples,
        { grid: createFilledGrid(prev.rows, prev.cols, -1), y: 1 },
      ],
    }));
  };

  const removeCustomSample = (index: number) => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.filter((_, idx) => idx !== index),
    }));
  };

  const toggleCustomCell = (sampleIndex: number, row: number, col: number) => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.map((sample, idx) => {
        if (idx !== sampleIndex) return sample;
        return {
          ...sample,
          grid: makeGrid(prev.rows, prev.cols, (r, c) => {
            if (r !== row || c !== col) return sample.grid[r][c];
            return sample.grid[r][c] === 1 ? -1 : 1;
          }),
        };
      }),
    }));
  };

  const updateSampleLabel = (index: number, value: number) => {
    setState((prev) => ({ ...prev, customApplied: false }));
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.map((sample, idx) =>
        idx === index ? { ...sample, y: value } : sample,
      ),
    }));
  };

  const applyCustomDataset = () => {
    setState((prev) => ({ ...prev, datasetName: "custom", gridRows: customConfig.rows, gridCols: customConfig.cols }));
    setCustomModalOpen(false);
    void reset({ ...apiConfig, customConfig }, "custom", true);
  };

  useEffect(() => {
    void reset(apiConfig, state.datasetName);
  }, []);


  useEffect(() => {
    if (!canFetchDiagnostics) return;
    if (dim !== 2) return;
    void diagnostics.fetchErrorSurface(errorSurfaceConfig);
  }, [canFetchDiagnostics, diagnostics.fetchErrorSurface, errorSurfaceConfig, dim]);

  useEffect(() => {
    if (!canFetchDiagnostics) return;
    void diagnostics.fetchMlpInternals(mlpConfig);
  }, [canFetchDiagnostics, diagnostics.fetchMlpInternals, mlpConfig]);

  useEffect(() => {
    setMlpSampleIndex(0);
  }, [state.datasetName, state.gridRows, state.gridCols]);

  return (
    <div id="app">
      <Header
        datasetName={state.datasetName}
        gridRows={state.gridRows}
        gridCols={state.gridCols}
        lr={lr}
        showCustomButton={state.datasetName === "custom"}
        route={route}
        showTrainingControls={showTrainingControls}
        onDatasetChange={handleDatasetChange}
        onStep={handleStep}
        onReset={handleReset}
        onLrChange={setLr}
        onOpenCustom={() => setCustomModalOpen(true)}
        onRouteChange={(next) => {
          navigate(next === "diagnostics" ? "/diagnostics" : next === "lms" ? "/lms" : "/");
        }}
      />

      <main className="grid">
        {route === "main" ? (
          <>
            <BoundaryPanel
              w={state.w}
              b={state.b}
              points={points}
              show={showBoundary}
              canvasRef={plotRef}
            />

            <SwitchboardPanel
              gridRows={state.gridRows}
              gridCols={state.gridCols}
              inputGrid={inputGrid}
              weightsBefore={weightsBefore}
              contribGrid={contribGrid}
              biasBefore={biasBefore}
              displayScore={displayScore}
              inputRef={inputRef}
              weightsBeforeRef={weightsBeforeRef}
              contribRef={contribRef}
              tooltip={tooltip}
              onTooltipChange={setTooltip}
              onTooltipHide={() => setTooltip((prev) => ({ ...prev, visible: false }))}
            />

            <AfterUpdatePanel
              gridRows={state.gridRows}
              gridCols={state.gridCols}
              weightsAfter={weightsAfter}
              biasAfter={state.b}
              lastDeltaW={state.lastStep?.deltaW ?? null}
              weightsAfterRef={weightsAfterRef}
              onTooltipChange={setTooltip}
              onTooltipHide={() => setTooltip((prev) => ({ ...prev, visible: false }))}
            />

            <StepMathPanel
              datasetName={state.datasetName}
              w={state.w}
              b={state.b}
              sampleIdx={state.sampleIdx}
              nextInput={state.nextInput}
              fallbackInput={fallbackInput}
              lastStep={state.lastStep}
              apiError={state.apiError}
            />

            <ExplanationPanel />
          </>
        ) : route === "diagnostics" ? (
          <>
            <ErrorSurfacePanel
              data={diagnostics.errorSurface}
              loading={diagnostics.errorSurfaceLoading}
              error={diagnostics.errorSurfaceError}
              config={errorSurfaceConfig}
              canShow={dim === 2}
              canFetch={canFetchDiagnostics}
              onConfigChange={setErrorSurfaceConfig}
            />

            <MlpInternalsPanel
              data={diagnostics.mlpInternals}
              loading={diagnostics.mlpInternalsLoading}
              error={diagnostics.mlpInternalsError}
              config={mlpConfig}
              canFetch={canFetchDiagnostics}
              onConfigChange={(next) => {
                setMlpHiddenDim(next.hiddenDim);
                setMlpSampleIndex(next.sampleIndex);
                setMlpLr(next.lr);
                setMlpSeed(next.seed);
              }}
            />
          </>
        ) : (
          <LmsPage apiBase={apiBase} />
        )}
      </main>

      <CustomModal
        isOpen={customModalOpen && state.datasetName === "custom"}
        customConfig={customConfig}
        onClose={() => setCustomModalOpen(false)}
        onRowsChange={updateCustomRows}
        onColsChange={updateCustomCols}
        onAddSample={addCustomSample}
        onApply={applyCustomDataset}
        onRemoveSample={removeCustomSample}
        onToggleCell={toggleCustomCell}
        onLabelChange={updateSampleLabel}
      />
    </div>
  );
}
