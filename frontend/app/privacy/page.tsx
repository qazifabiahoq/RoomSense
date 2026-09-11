import Link from "next/link";

export const metadata = {
  title: "Privacy Policy - RoomSense",
};

export default function PrivacyPage() {
  return (
    <main className="mx-auto max-w-2xl px-5 py-12 sm:py-16">
      <Link href="/" className="text-sm text-neutral-500 underline decoration-neutral-300 underline-offset-2 hover:text-neutral-800">
        Back to RoomSense
      </Link>

      <h1 className="font-display mt-4 text-2xl font-bold text-neutral-900 sm:text-3xl">Privacy Policy</h1>
      <p className="mt-1 text-sm text-neutral-400">Last updated: September 2026</p>

      <div className="mt-8 space-y-6 text-sm leading-relaxed text-neutral-700">
        <p>
          RoomSense is built to work without accounts, sign ups, or tracking. This page explains what
          happens to a photo when you use it.
        </p>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Photos you upload or capture</h2>
          <p className="mt-2">
            When you upload a photo or take one with your camera, it is sent to RoomSense's backend for
            analysis (object detection, lighting, and color extraction) and held in memory only for the
            length of that request. It is not written to a database or disk, and it is not kept once the
            response is sent back to your browser. If you request an AI redesign, your photo and room type
            are sent to a third party generative image service to produce the result; that request is
            subject to that service's own handling of the data it receives.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Camera access</h2>
          <p className="mt-2">
            The live camera feature uses your browser's camera permission to show a live preview and let
            you capture a single frame. The camera stream is never recorded and never leaves your device
            until you press the capture button, at which point only the single captured photo is sent for
            analysis, the same as an uploaded file.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Accounts and personal data</h2>
          <p className="mt-2">
            RoomSense does not require an account, does not ask for your name, email, or location, and does
            not use cookies to track you across sites.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Hosting and infrastructure logs</h2>
          <p className="mt-2">
            Like most web services, the hosting providers behind RoomSense (Vercel for the interface, Render
            for the analysis backend) may keep short lived, standard operational logs, such as request
            timestamps and error logs, for reliability and security purposes. These are the providers'
            infrastructure logs, not something RoomSense adds on top, and they are not used to build a
            profile of you.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Third party services</h2>
          <p className="mt-2">
            RoomSense uses Pollinations.ai for AI image generation. When you request a redesign, your photo
            and chosen style are sent to that service to produce the result. Review Pollinations.ai's own
            policies if you want details on how they handle requests they receive.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Children's privacy</h2>
          <p className="mt-2">
            RoomSense is not directed at children and does not knowingly collect data from them.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Changes to this policy</h2>
          <p className="mt-2">
            This policy may be updated as the app changes. The date at the top of this page reflects the
            most recent update.
          </p>
        </section>

        <section>
          <h2 className="font-display text-base font-semibold text-neutral-900">Contact</h2>
          <p className="mt-2">Questions about this policy can be raised as an issue on the RoomSense GitHub repository.</p>
        </section>
      </div>
    </main>
  );
}
