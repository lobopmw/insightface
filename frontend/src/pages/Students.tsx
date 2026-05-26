import { GraduationCap, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { StatCard } from "@/components/dashboard/StatCard";
import { EmbeddingStatusCard } from "@/components/students/EmbeddingStatusCard";
import { FaceCaptureWizard } from "@/components/students/FaceCaptureWizard";
import { StudentForm } from "@/components/students/StudentForm";
import { StudentProfileCard } from "@/components/students/StudentProfileCard";
import { StudentTable } from "@/components/students/StudentTable";
import {
  createStudent,
  deactivateStudent,
  generateStudentEmbeddings,
  uploadStudentFaceImage,
} from "@/services/studentsApi";
import { useNextRegistration, useStudentClasses, useStudentFaceStatus, useStudents } from "@/hooks/useStudents";
import { useAuthStore } from "@/stores/authStore";
import type { PoseKey, Student, StudentCreatePayload } from "@/types/student";

export function Students() {
  const token = useAuthStore((state) => state.token);
  const queryClient = useQueryClient();
  const { data, isLoading, isError } = useStudents();
  const { data: classesData } = useStudentClasses();
  const { data: nextRegistration, refetch: refetchNextRegistration } = useNextRegistration();
  const students = data?.items ?? [];
  const classes = classesData?.items ?? [];
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [embeddingMessage, setEmbeddingMessage] = useState<string | null>(null);
  const { data: faceStatus, refetch: refetchFaceStatus } = useStudentFaceStatus(selectedStudent?.id);

  useEffect(() => {
    if (!selectedStudent && students[0]) {
      setSelectedStudent(students[0]);
    }
  }, [selectedStudent, students]);

  const createMutation = useMutation({
    mutationFn: (payload: StudentCreatePayload) => createStudent(token as string, payload),
    onSuccess: (student) => {
      setSelectedStudent(student);
      setEmbeddingMessage(null);
      void queryClient.invalidateQueries({ queryKey: ["students"] });
      void refetchNextRegistration();
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (studentId: string) => deactivateStudent(token as string, studentId),
    onSuccess: () => {
      setSelectedStudent(null);
      void queryClient.invalidateQueries({ queryKey: ["students"] });
    },
  });

  const uploadSingleImage = async (pose: PoseKey, image: Blob, index: number) => {
    if (!token || !selectedStudent?.id) {
      return;
    }
    const response = await uploadStudentFaceImage(token, selectedStudent.id, pose, image, index);
    setEmbeddingMessage(`Imagem ${index}/10 registrada para ${pose}.`);
    queryClient.setQueryData(["students", selectedStudent.id, "face-status"], response.status);
    void refetchFaceStatus();
  };

  const embeddingMutation = useMutation({
    mutationFn: () => generateStudentEmbeddings(token as string, selectedStudent?.id as string),
    onSuccess: (response) => {
      setEmbeddingMessage(response.message);
      void refetchFaceStatus();
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

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 rounded-lg border border-border bg-card p-5 shadow-sm md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm font-medium text-primary">Cadastro facial</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">Alunos</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Cadastro acadêmico, captura por poses e geração de embeddings para reconhecimento.
          </p>
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-2">
        <StatCard helper="Registros carregados da API" icon={Users} label="Alunos listados" tone="blue" value={isLoading ? "..." : students.length} />
        <StatCard helper="Turmas disponiveis para associacao" icon={GraduationCap} label="Turmas" tone="green" value={classes.length} />
      </section>

      {isError ? <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">Nao foi possivel carregar alunos.</p> : null}

      <section className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <StudentForm
          classes={classes}
          defaultRegistration={nextRegistration?.matricula}
          isLoading={createMutation.isPending}
          onSubmit={(payload) => createMutation.mutate(payload)}
        />
        <StudentProfileCard status={faceStatus} student={selectedStudent} />
      </section>

      {createMutation.isError ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">Nao foi possivel salvar o aluno.</p>
      ) : null}

      <FaceCaptureWizard
        onUploadImage={uploadSingleImage}
        selectedStudent={selectedStudent}
        status={faceStatus}
      />

      <EmbeddingStatusCard
        canGenerate={canGenerateEmbeddings}
        isLoading={embeddingMutation.isPending}
        message={embeddingMessage}
        onGenerate={() => embeddingMutation.mutate()}
        status={faceStatus}
      />

      <StudentTable
        isLoading={isLoading}
        onDelete={(studentId) => deactivateMutation.mutate(studentId)}
        onSelect={(student) => {
          setSelectedStudent(student);
          setEmbeddingMessage(null);
        }}
        students={students}
      />
    </div>
  );
}
