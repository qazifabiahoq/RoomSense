const PHOTOS = [
  { url: "https://images.unsplash.com/photo-1556912173-3bb406ef7e77?w=500&h=340&fit=crop&q=80", caption: "Modern and airy" },
  { url: "https://images.unsplash.com/photo-1505691938895-1758d7feb511?w=500&h=340&fit=crop&q=80", caption: "Warm and cozy" },
];

export default function InspirationGallery({ roomType }: { roomType: string }) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-5 shadow-sm sm:p-6">
      <h3 className="font-display mb-1 text-base font-semibold text-neutral-900">Get inspired</h3>
      <p className="mb-4 text-sm text-neutral-500">A couple of {roomType.toLowerCase()} looks to spark ideas.</p>
      <div className="grid grid-cols-2 gap-3">
        {PHOTOS.map((photo) => (
          <div key={photo.url} className="overflow-hidden rounded-lg border border-neutral-200">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={photo.url} alt={photo.caption} className="h-32 w-full object-cover sm:h-40" />
            <p className="px-2.5 py-2 text-xs font-medium text-neutral-600">{photo.caption}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
