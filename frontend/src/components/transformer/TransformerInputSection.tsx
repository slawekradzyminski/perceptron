type TransformerInputSectionProps = {
  inputText: string;
  onInputChange: (value: string) => void;
};

export function TransformerInputSection({
  inputText,
  onInputChange,
}: TransformerInputSectionProps) {
  return (
    <div className="transformer-input">
      <label>Enter text:</label>
      <textarea
        value={inputText}
        onChange={(e) => onInputChange(e.target.value)}
        placeholder="Enter text to tokenize..."
        rows={3}
      />
    </div>
  );
}
