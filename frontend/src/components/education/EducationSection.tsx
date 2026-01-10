import { ReactNode } from "react";

type EducationSectionProps = {
  title: string;
  children: ReactNode;
};

export function EducationSection({ title, children }: EducationSectionProps) {
  return (
    <div className="education-section">
      <h4>{title}</h4>
      {children}
    </div>
  );
}

type EducationPanelProps = {
  title: string;
  children: ReactNode;
  className?: string;
};

export function EducationPanel({ title, children, className = "" }: EducationPanelProps) {
  return (
    <div className={`education-panel ${className}`}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}
