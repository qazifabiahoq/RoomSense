"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export default function UploadZone({
  onFileSelected,
  disabled,
}: {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [dragging, setDragging] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.type.startsWith("image/")) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
    setStarting(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const openCamera = useCallback(async () => {
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Live camera isn't supported in this browser. Please upload a photo instead.");
      return;
    }

    setStarting(true);
    setCameraOpen(true);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
    } catch {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch {
        setCameraError("Couldn't access your camera. Check your browser's camera permission and try again.");
        setCameraOpen(false);
        setStarting(false);
        return;
      }
    }

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => {});
    }
    setStarting(false);
  }, []);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `room-photo-${Date.now()}.jpg`, { type: "image/jpeg" });
        stopCamera();
        onFileSelected(file);
      },
      "image/jpeg",
      0.92
    );
  }, [onFileSelected, stopCamera]);

  if (cameraOpen) {
    return (
      <div className="overflow-hidden rounded-xl border border-neutral-200 bg-black">
        <div className="relative aspect-[4/3] w-full">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover" />
          {starting && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-white/80">
              Starting camera...
            </div>
          )}
        </div>
        <div className="flex items-center justify-center gap-3 bg-white p-3">
          <button
            onClick={stopCamera}
            className="rounded-lg px-4 py-2 text-sm font-medium text-neutral-600 hover:bg-neutral-100"
          >
            Cancel
          </button>
          <button
            onClick={capturePhoto}
            disabled={starting}
            className="rounded-lg bg-brand-500 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
          >
            Take photo
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
          dragging ? "border-brand-400 bg-brand-50" : "border-neutral-200 bg-neutral-50/50 hover:border-brand-300"
        } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      >
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="mb-1 text-brand-500">
          <path d="M12 16V4M12 4L7 9M12 4l5 5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <p className="font-medium text-neutral-800">Drag a photo here, or click to choose one</p>
        <p className="text-sm text-neutral-400">JPG, PNG, or WEBP</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <button
        type="button"
        onClick={openCamera}
        disabled={disabled}
        className="flex w-full items-center justify-center gap-2 rounded-xl border border-neutral-200 bg-white px-4 py-3 text-sm font-medium text-neutral-700 hover:border-brand-300 hover:text-brand-600 disabled:pointer-events-none disabled:opacity-60"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path
            d="M4 8a2 2 0 012-2h1.2l.9-1.5A2 2 0 019.8 3.5h4.4a2 2 0 011.7 1l.9 1.5H18a2 2 0 012 2v9a2 2 0 01-2 2H6a2 2 0 01-2-2V8z"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="12.5" r="3.2" stroke="currentColor" strokeWidth="1.7" />
        </svg>
        Take a live photo
      </button>

      {cameraError && <p className="text-center text-sm text-red-600">{cameraError}</p>}
    </div>
  );
}
