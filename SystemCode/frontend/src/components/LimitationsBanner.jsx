export default function LimitationsBanner({ items }) {
  return (
    <div className="bg-amber-50 border-b border-amber-300" role="note" aria-label="limitations">
      <div className="max-w-6xl mx-auto px-4 py-2 text-sm text-amber-900">
        <strong>Research demonstration — development results, not validated for real machines.</strong>
        {items && items.length > 1 && (
          <details className="inline ml-2">
            <summary className="inline cursor-pointer underline">All limitations ({items.length})</summary>
            <ul className="list-disc ml-6 mt-1">{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
          </details>
        )}
      </div>
    </div>
  );
}
