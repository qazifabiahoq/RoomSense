import { RoomType, StyleConfig, Zone } from "./types";

export const ROOM_TYPES: RoomType[] = [
  "Living Room",
  "Bedroom",
  "Kitchen",
  "Bathroom",
  "Dining Room",
  "Home Office",
  "Kids Room",
  "Laundry Room",
];

export const ROOM_CONFIGS: Record<RoomType, Zone[]> = {
  "Living Room": [
    {
      name: "Seating Area",
      location: "Center of room, facing TV or focal point",
      furniture: ["3-Seater Sofa", "Accent Chairs (2x)", "Coffee Table", "Side Tables", "Floor Lamp", "Area Rug"],
      lighting: "Layered: Overhead pendant + Floor lamp + Table lamps (2700-3000K)",
      considerations: [
        "Leave 45cm walking space around furniture",
        "Position sofa 2-3m from TV",
        "Create conversation pit with chairs facing each other",
      ],
    },
    {
      name: "Entertainment Zone",
      location: "Against main wall",
      furniture: ["TV Stand or Media Console", "Wall-mounted TV", "Cable Management Box", "Sound Bar", "Storage Baskets"],
      lighting: "LED bias lighting behind TV",
      considerations: [
        "Mount TV at eye level when seated",
        "Hide cables with cable covers",
        "Add closed storage for media clutter",
      ],
    },
    {
      name: "Reading Nook",
      location: "Corner near window",
      furniture: ["Comfortable Armchair", "Reading Lamp", "Small Side Table", "Throw Blanket", "Bookshelf"],
      lighting: "Adjustable task lamp (reading light)",
      considerations: [
        "Position near natural light source",
        "Add floor cushion for flexibility",
        "Keep books within arm's reach",
      ],
    },
  ],
  Bedroom: [
    {
      name: "Sleeping Area",
      location: "Against longest wall, away from door",
      furniture: ["Bed Frame", "Mattress", "Nightstands (2x)", "Table Lamps (2x)", "Headboard"],
      lighting: "Bedside lamps with dimmer switches (2700K warm)",
      considerations: [
        "Allow 60cm on each side of bed",
        "Position bed away from direct sunlight",
        "Avoid placing bed under window",
      ],
    },
    {
      name: "Storage & Dressing",
      location: "Opposite or adjacent to bed",
      furniture: ["Wardrobe or Closet System", "Dresser with Mirror", "Clothing Rack", "Storage Boxes", "Bench"],
      lighting: "Overhead lighting + Mirror lights",
      considerations: [
        "Keep wardrobe doors clearance 90cm",
        "Use vertical space efficiently",
        "Add drawer organizers",
      ],
    },
    {
      name: "Personal Space",
      location: "Corner or window area",
      furniture: ["Accent Chair", "Small Desk or Vanity", "Ottoman", "Full-length Mirror"],
      lighting: "Task lighting for vanity area",
      considerations: ["Create relaxation spot", "Add plants for air quality", "Keep surfaces minimal"],
    },
  ],
  Kitchen: [
    {
      name: "Cooking Zone",
      location: "Stove, counter, sink triangle",
      furniture: ["Kitchen Island or Cart", "Bar Stools (2-3x)", "Pot Rack", "Spice Rack", "Cutting Board Station"],
      lighting: "Under-cabinet LED strips + Pendant lights over island",
      considerations: [
        "Keep 120cm between counters",
        "Place frequently used items within reach",
        "Add anti-fatigue mat",
      ],
    },
    {
      name: "Storage & Pantry",
      location: "Along walls, maximize vertical space",
      furniture: ["Pantry Shelving", "Upper Cabinets", "Pull-out Drawers", "Lazy Susan", "Clear Storage Containers"],
      lighting: "Interior cabinet lights",
      considerations: ["Group items by category", "Use clear containers for visibility", "Label everything"],
    },
    {
      name: "Dining/Eating Area",
      location: "Adjacent to kitchen",
      furniture: ["Dining Table", "Dining Chairs", "Pendant Light", "Buffet or Sideboard"],
      lighting: "Statement pendant 75cm above table",
      considerations: [
        "Allow 60cm per person at table",
        "Leave 90cm walking clearance",
        "Add rug under table for comfort",
      ],
    },
  ],
  Bathroom: [
    {
      name: "Vanity Area",
      location: "Primary wall space",
      furniture: ["Vanity with Sink", "Mirror (large)", "Wall-mounted Shelves", "Toiletry Organizers", "Towel Bar"],
      lighting: "Side-mounted mirror lights + Overhead (4000K)",
      considerations: [
        "Install lighting at face level, not overhead",
        "Add storage for daily items",
        "Keep counter clutter-free",
      ],
    },
    {
      name: "Shower/Bath Zone",
      location: "Wet area with proper drainage",
      furniture: ["Shower Caddy", "Bath Mat", "Towel Hooks", "Shower Curtain or Glass Door"],
      lighting: "Waterproof recessed lighting",
      considerations: ["Use non-slip mats", "Add grab bar for safety", "Ensure proper ventilation"],
    },
    {
      name: "Storage Solutions",
      location: "Walls, over toilet, under sink",
      furniture: ["Over-toilet Cabinet", "Under-sink Organizers", "Medicine Cabinet", "Towel Ladder", "Baskets"],
      lighting: "Ambient ceiling light",
      considerations: [
        "Use vertical wall space",
        "Keep cleaning supplies accessible",
        "Store towels within reach",
      ],
    },
  ],
  "Dining Room": [
    {
      name: "Main Dining Area",
      location: "Center of room",
      furniture: ["Dining Table (6-8 seater)", "Dining Chairs", "Table Runner", "Centerpiece", "Area Rug"],
      lighting: "Statement chandelier or pendant (centered, 75-85cm above table)",
      considerations: [
        "Allow 60cm per person",
        "Leave 90-120cm walking space around table",
        "Rug should extend 60cm beyond table edges",
      ],
    },
    {
      name: "Serving Station",
      location: "Against wall, near kitchen",
      furniture: ["Buffet or Sideboard", "Table Lamp", "Serving Trays", "Wine Rack", "Storage for Linens"],
      lighting: "Accent lighting with table lamps",
      considerations: ["Height should be 75-90cm", "Use for dish storage and serving", "Add decorative items on top"],
    },
    {
      name: "Display Area",
      location: "Open wall space",
      furniture: ["China Cabinet", "Display Shelves", "Artwork", "Mirror"],
      lighting: "Picture lights or spotlights",
      considerations: ["Showcase special dinnerware", "Create visual interest", "Balance with room size"],
    },
  ],
  "Home Office": [
    {
      name: "Work Station",
      location: "Near natural light, against wall",
      furniture: ["Desk (140x70cm)", "Ergonomic Office Chair", "Monitor Stand", "Desk Lamp", "Cable Management"],
      lighting: "Task lamp + Ambient overhead (4000-5000K)",
      considerations: [
        "Position desk perpendicular to window",
        "Monitor 50-70cm from eyes",
        "Add footrest if needed",
      ],
    },
    {
      name: "Storage & Filing",
      location: "Adjacent to desk, within reach",
      furniture: ["Filing Cabinet", "Bookshelf", "Storage Boxes", "Magazine Holders", "Printer Stand"],
      lighting: "Overhead lighting",
      considerations: ["Keep frequently used items accessible", "Use vertical storage", "Label all files clearly"],
    },
    {
      name: "Meeting/Reading Corner",
      location: "Opposite desk area",
      furniture: ["Comfortable Chair", "Small Side Table", "Bookshelf", "Floor Lamp"],
      lighting: "Adjustable reading lamp",
      considerations: ["Create separation from work desk", "Add plants for relaxation", "Use for video calls"],
    },
  ],
  "Kids Room": [
    {
      name: "Sleep Zone",
      location: "Quiet corner, away from play area",
      furniture: ["Bed with Storage", "Nightstand", "Night Light", "Blackout Curtains"],
      lighting: "Dimmable ceiling light + Night light",
      considerations: [
        "Use bed rails for young children",
        "Keep pathway clear",
        "Add comfort items (pillows, stuffed animals)",
      ],
    },
    {
      name: "Play & Activity Area",
      location: "Open floor space, center of room",
      furniture: ["Toy Storage Bins", "Play Mat", "Small Table & Chairs", "Toy Organizer", "Bookshelf"],
      lighting: "Bright overhead lighting",
      considerations: [
        "Use low storage for easy access",
        "Rotate toys regularly",
        "Create designated zones for activities",
      ],
    },
    {
      name: "Study Corner",
      location: "Near window, quiet area",
      furniture: ["Kid-sized Desk", "Adjustable Chair", "Desk Lamp", "Supply Organizer", "Bulletin Board"],
      lighting: "Task lighting for homework",
      considerations: [
        "Adjust furniture as child grows",
        "Keep supplies organized",
        "Display artwork and achievements",
      ],
    },
  ],
  "Laundry Room": [
    {
      name: "Washing Station",
      location: "Against wall with plumbing",
      furniture: ["Washer & Dryer", "Laundry Baskets (3x for sorting)", "Hamper", "Rolling Cart"],
      lighting: "Bright overhead LED (4000K)",
      considerations: ["Leave 10cm space behind machines", "Use vibration pads", "Sort lights, darks, delicates"],
    },
    {
      name: "Folding & Ironing",
      location: "Open counter space",
      furniture: ["Folding Counter", "Wall-mounted Ironing Board", "Iron Holder", "Drying Rack", "Shelf for Detergent"],
      lighting: "Under-cabinet lights",
      considerations: ["Counter height 85-90cm", "Keep iron at safe distance", "Add cushioned mat for standing"],
    },
    {
      name: "Storage & Organization",
      location: "Upper cabinets and shelving",
      furniture: ["Upper Cabinets", "Shelving Units", "Clear Storage Jars", "Hanging Rod", "Utility Sink"],
      lighting: "General ambient lighting",
      considerations: [
        "Store detergents out of reach of children",
        "Label all products",
        "Keep stain removers accessible",
      ],
    },
  ],
};

export const REDESIGN_STYLES: Record<string, StyleConfig> = {
  "Modern Minimalist": {
    description: "Clean Scandinavian aesthetic with sleek furniture, neutral tones, and open spaces",
    colors: ["#FFFFFF", "#F5F5F5", "#E0E0E0", "#757575"],
    prompt:
      "modern minimalist interior design, sleek contemporary furniture, clean white walls, scandinavian style, bright natural light, open space, professional interior photography, 8k uhd",
    negativePrompt: "cluttered, messy, dark, ornate, traditional, busy patterns, low quality, blurry",
  },
  "Cozy Traditional": {
    description: "Warm, inviting spaces with classic furniture, rich textures, and comfortable seating",
    colors: ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3"],
    prompt:
      "cozy traditional interior design, classic comfortable furniture, warm wood tones, soft textiles, warm lighting, inviting atmosphere, professional interior photography, 8k uhd",
    negativePrompt: "modern, minimalist, cold, sterile, empty, stark, low quality, blurry",
  },
};

export function paletteSuggestions(roomType: RoomType): { name: string; colors: string[] }[] {
  if (roomType === "Bedroom" || roomType === "Kids Room") {
    return [
      { name: "Calm & Serene", colors: ["#E8EAF6", "#C5CAE9", "#9FA8DA", "#7986CB"] },
      { name: "Warm & Cozy", colors: ["#FFF3E0", "#FFE0B2", "#FFCC80", "#FFB74D"] },
      { name: "Modern Neutral", colors: ["#FAFAFA", "#EEEEEE", "#BDBDBD", "#757575"] },
    ];
  }
  if (roomType === "Home Office") {
    return [
      { name: "Focus Blue", colors: ["#E3F2FD", "#BBDEFB", "#90CAF9", "#42A5F5"] },
      { name: "Professional Grey", colors: ["#FAFAFA", "#ECEFF1", "#B0BEC5", "#546E7A"] },
      { name: "Energizing Green", colors: ["#E8F5E9", "#C8E6C9", "#81C784", "#66BB6A"] },
    ];
  }
  if (roomType === "Living Room" || roomType === "Dining Room") {
    return [
      { name: "Welcoming Warm", colors: ["#FFF8E1", "#FFECB3", "#FFD54F", "#FFA726"] },
      { name: "Elegant Neutral", colors: ["#F5F5F5", "#E0E0E0", "#9E9E9E", "#616161"] },
      { name: "Fresh Modern", colors: ["#E0F2F1", "#B2DFDB", "#4DB6AC", "#26A69A"] },
    ];
  }
  if (roomType === "Kitchen") {
    return [
      { name: "Clean White", colors: ["#FFFFFF", "#F8F9FA", "#E9ECEF", "#CED4DA"] },
      { name: "Classic Wood Tones", colors: ["#F5E6D3", "#D7C9B8", "#A89784", "#8B7355"] },
      { name: "Modern Charcoal", colors: ["#F5F5F5", "#E0E0E0", "#757575", "#424242"] },
    ];
  }
  if (roomType === "Bathroom") {
    return [
      { name: "Spa Blue", colors: ["#E1F5FE", "#B3E5FC", "#4FC3F7", "#0288D1"] },
      { name: "Fresh White", colors: ["#FFFFFF", "#F5F5F5", "#EEEEEE", "#BDBDBD"] },
      { name: "Warm Beige", colors: ["#FFF8E1", "#FFECB3", "#FFD54F", "#F9A825"] },
    ];
  }
  return [
    { name: "Bright & Airy", colors: ["#FFFFFF", "#F5F5F5", "#EEEEEE", "#E0E0E0"] },
    { name: "Warm Neutral", colors: ["#FBE9E7", "#FFCCBC", "#FF8A65", "#FF7043"] },
    { name: "Cool Modern", colors: ["#E1F5FE", "#B3E5FC", "#4FC3F7", "#29B6F6"] },
  ];
}
