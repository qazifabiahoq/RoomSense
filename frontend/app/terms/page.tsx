import Link from "next/link";

export const metadata = {
  title: "Terms of Service - RoomSense",
};

export default function TermsPage() {
  return (
    <main className="mx-auto max-w-2xl px-5 py-12 sm:py-16">
      <Link href="/" className="text-sm text-neutral-500 underline decoration-neutral-300 underline-offset-2 hover:text-neutral-800">
        Back to RoomSense
      </Link>

      <h1 className="font-display mt-4 text-2xl font-bold text-neutral-900 sm:text-3xl">Terms of Service</h1>
      <p className="mt-1 text-sm text-neutral-400">Last updated: September 2026</p>

      <div className="mt-8 space-y-6 text-sm leading-relaxed text-neutral-700">
        <p>
          By using RoomSense, you agree to these terms. If you do not agree, please do not use the app.
        </p>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">What RoomSense is</h2>
          <p className="mt-2">
            RoomSense analyzes a photo of a room you upload or capture and returns furniture detection,
            lighting and color information, curated design suggestions, and an optional AI generated
            redesign image. It is a design inspiration tool, not a professional service.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Not professional advice</h2>
          <p className="mt-2">
            Nothing RoomSense produces is architectural, engineering, electrical, structural, or contracting
            advice, and it is not a substitute for a licensed professional. Before making structural changes,
            electrical or plumbing changes, or any purchase or renovation decision, consult a qualified
            professional and follow local building codes and regulations. You are solely responsible for
            decisions you make based on RoomSense's output, including furniture purchases, room layout
            changes, and renovations.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">AI and automated output</h2>
          <p className="mt-2">
            Object detection, lighting readings, and color palettes are produced by automated models and
            image processing and may be incomplete or inaccurate. The redesigned room image is produced by
            a third party generative AI service and is an artistic interpretation, not a guarantee of how
            your space will look after any purchase or renovation. Curated design recommendations come from
            a general knowledge base and are not tailored to your specific building, budget, or local codes.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Your photos</h2>
          <p className="mt-2">
            You are responsible for the photos you upload or capture. Do not submit a photo of a room or
            space you do not have the right to share, or one containing other identifiable people who have
            not agreed to be photographed. See the{" "}
            <Link href="/privacy" className="underline decoration-neutral-300 underline-offset-2 hover:text-neutral-900">
              Privacy Policy
            </Link>{" "}
            for how photos are handled.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Third party services</h2>
          <p className="mt-2">
            RoomSense sends your photo to a backend service for analysis and, if you request a redesign,
            to a third party generative image service to produce the result. Those services operate under
            their own terms, which are outside RoomSense's control.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">No warranty</h2>
          <p className="mt-2">
            RoomSense is provided as is and as available, without warranties of any kind, whether express
            or implied, including accuracy, reliability, or fitness for a particular purpose. The app may
            be unavailable, slow, or produce incorrect results at times.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Limitation of liability</h2>
          <p className="mt-2">
            To the fullest extent permitted by law, RoomSense and its creator are not liable for any
            indirect, incidental, or consequential damages, or for any loss, injury, or cost arising from
            your use of the app or from decisions made based on its output, including furniture purchases,
            renovations, or structural changes.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Age</h2>
          <p className="mt-2">
            RoomSense does not knowingly collect information from children. If you are under the age
            required by the laws of your country to use online services without parental consent, please
            use the app only with a parent or guardian.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Changes to these terms</h2>
          <p className="mt-2">
            These terms may be updated from time to time. Continued use of RoomSense after a change means
            you accept the updated terms.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Contact</h2>
          <p className="mt-2">Questions about these terms can be raised as an issue on the RoomSense GitHub repository.</p>
        </section>
      </div>
    </main>
  );
}
