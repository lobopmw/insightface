import { Camera, ListChecks, Sparkles, UserRound } from "lucide-react";
import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { EmbeddingStatusCard } from "@/components/students/EmbeddingStatusCard";
import { FaceCaptureWizard } from "@/components/students/FaceCaptureWizard";
import { StudentForm } from "@/components/students/StudentForm";
import { StudentTable } from "@/components/students/StudentTable";
import {
  createStudent,
  deactivateStudent,
  generateStudentEmbeddings,
  uploadStudentFaceImage,
} from "@/services/studentsApi";
import {
  useNextRegistration,
  useStudentClasses,
  useStudentFaceStatus,
  useStudents,
  useStudentsFaceStatuses,
} from "@/hooks/useStudents";
import { useAuthStore } from "@/stores/authStore";
import type { PoseKey, Student, StudentCreatePayload } from "@/types/student";

type StudentStep = "identity" | "capture" | "embeddings" | "list";

export function Students() {
  const token = useAuthStore((state) => state.token);
  const queryClient = useQueryClient();
  const { data, isLoading, isError } = useStudents();
  const { data: faceStatusesData } = useStudentsFaceStatuses();
  const { data: classesData } = useStudentClasses();
  const { data: nextRegistration, refetch: refetchNextRegistration } = useNextRegistration();
  const students = data?.items ?? [];
  const classes = classesData?.items ?? [];
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [embeddingMessage, setEmbeddingMessage] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState<StudentStep>("identity");
  const { data: faceStatus, refetch: refetchFaceStatus } = useStudentFaceStatus(selectedStudent?.id);
  const faceStatuses = useMemo(
    () =>
      Object.fromEntries(
        (faceStatusesData?.items ?? []).map((item) => [item.student_id, item]),
      ),
    [faceStatusesData],
  );

  const createMutation = useMutation({
    mutationFn: (payload: StudentCreatePayload) => createStudent(token as string, payload),
    onSuccess: (student) => {
      setSelectedStudent(student);
      setEmbeddingMessage(null);
      setActiveStep("capture");
      void queryClient.invalidateQueries({ queryKey: ["students"] });
      void queryClient.invalidateQueries({ queryKey: ["students", "face-status"] });
      void refetchNextRegistration();
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (studentId: string) => deactivateStudent(token as string, studentId),
    onSuccess: () => {
      setSelectedStudent(null);
      void queryClient.invalidateQueries({ queryKey: ["students"] });
      void queryClient.invalidateQueries({ queryKey: ["students", "face-status"] });
    },
  });

  const uploadSingleImage = async (pose: PoseKey, image: Blob, index: number) => {
    if (!token || !selectedStudent?.id) {
      return;
    }
    const response = await uploadStudentFaceImage(token, selectedStudent.id, pose, image, index);
    setEmbeddingMessage(`Imagem ${index}/10 registrada para ${pose}.`);
    queryClient.setQueryData(["students", selectedStudent.id, "face-status"], response.status);
    queryClient.setQueryData(["students", "face-status"], {
      items: Object.values({ ...faceStatuses, [selectedStudent.id]: response.status }),
    });
    void queryClient.invalidateQueries({ queryKey: ["students", "face-status"] });
    void refetchFaceStatus();
  };

  const embeddingMutation = useMutation({
    mutationFn: () => generateStudentEmbeddings(token as string, selectedStudent?.id as string),
    onSuccess: (response) => {
      setEmbeddingMessage(response.message);
      if (selectedStudent?.id) {
        queryClient.setQueryData(["students", selectedStudent.id, "face-status"], response.status);
        queryClient.setQueryData(["students", "face-status"], {
          items: Object.values({ ...faceStatuses, [selectedStudent.id]: response.status }),
        });
      }
      void queryClient.invalidateQueries({ queryKey: ["students", "face-status"] });
      void refetchFaceStatus();
      setActiveStep("list");
    },
    onError: (error) => {
      setEmbeddingMessage(error instanceof Error ? error.message : "Falha ao gerar embeddings.");
    },
  });

  const canGenerateEmbeddings = useMemo(
    () =>
      Boolean(
        selectedStudent &&
          faceStatus?.frontal.complete &&
          faceStatus?.lateral_esquerda.complete &&
          faceStatus?.lateral_direita.complete &&
          faceStatus?.cabeca_baixa.complete,
      ),
    [faceStatus, selectedStudent],
  );
  const allPosesComplete = canGenerateEmbeddings;
  const studentSteps: Array<{
    id: StudentStep;
    label: string;
    icon: typeof UserRound;
    enabled: boolean;
  }> = [
    { id: "identity", label: "Dados", icon: UserRound, enabled: true },
    { id: "capture", label: "Captura", icon: Camera, enabled: Boolean(selectedStudent) },
    { id: "embeddings", label: "Embeddings", icon: Sparkles, enabled: Boolean(selectedStudent && allPosesComplete) },
    { id: "list", label: "Lista", icon: ListChecks, enabled: true },
  ];

  return (
    <div className="space-y-6">
      {isError ? <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">Nao foi possivel carregar alunos.</p> : null}

      <section className="rounded-lg border border-border bg-card p-2 shadow-sm shadow-slate-200/60">
        <div className="flex gap-2 overflow-x-auto">
          {studentSteps.map((step, index) => {
            const Icon = step.icon;
            return (
              <button
                key={step.id}
                className={`flex h-10 shrink-0 items-center gap-2 rounded-md px-3 text-sm font-semibold transition ${
                  activeStep === step.id
                    ? "bg-slate-950 text-white"
                    : step.enabled
                      ? "bg-background text-muted-foreground hover:bg-muted hover:text-foreground"
                      : "cursor-not-allowed bg-slate-100 text-slate-400"
                }`}
                disabled={!step.enabled}
                onClick={() => setActiveStep(step.id)}
                type="button"
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-sm bg-white/15 text-xs">{index + 1}</span>
                <Icon className="h-4 w-4" />
                {step.label}
              </button>
            );
          })}
        </div>
      </section>

      {activeStep === "identity" ? (
        <StudentForm
          classes={classes}
          defaultRegistration={nextRegistration?.matricula}
          isLoading={createMutation.isPending}
          onSelectExisting={(student) => {
            setSelectedStudent(student);
            setEmbeddingMessage(null);
            setActiveStep("capture");
          }}
          onSubmit={(payload) => createMutation.mutate(payload)}
          selectedStudent={selectedStudent}
          students={students}
        />
      ) : null}

      {createMutation.isError ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">Nao foi possivel salvar o aluno.</p>
      ) : null}

      {activeStep === "capture" ? (
        <FaceCaptureWizard
          onUploadImage={uploadSingleImage}
          selectedStudent={selectedStudent}
          status={faceStatus}
        />
      ) : null}

      {activeStep === "embeddings" ? (
        <EmbeddingStatusCard
          canGenerate={canGenerateEmbeddings}
          isLoading={embeddingMutation.isPending}
          message={embeddingMessage}
          onGenerate={() => embeddingMutation.mutate()}
          status={faceStatus}
        />
      ) : null}

      {activeStep === "list" ? (
        <StudentTable
          isLoading={isLoading}
          onDelete={(studentId) => deactivateMutation.mutate(studentId)}
          onSelect={(student) => {
            setSelectedStudent(student);
            setEmbeddingMessage(null);
            setActiveStep("capture");
          }}
          selectedStudentId={selectedStudent?.id}
          statuses={faceStatuses}
          students={students}
        />
      ) : null}
    </div>
  );
}
