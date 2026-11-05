"""Synthetic source texts for the sample dataset.

All text is written for this project (no proprietary / private / scraped data).
Fictional companies only.  The 'LLM-generated' sample is deliberately written in
a generic chat-assistant style to exercise the AI-likelihood signal.
"""

TEXT_A = (
    "Northwind Foods should enter the Vietnamese snack market through a joint venture "
    "rather than a wholly owned subsidiary. Our analysis rests on three observations. "
    "First, the distribution network in Vietnam is fragmented: roughly 70% of snack sales "
    "pass through small family-run shops that no national chain reaches. A local partner "
    "already has those relationships, and building them from scratch would take us four to "
    "five years. Second, regulatory approval for imported flavourings can take up to eleven "
    "months, whereas a partner with an existing licence cuts that to about three. Third, "
    "consumers there prefer sweet-and-salty flavours that our current range does not offer, "
    "so product adaptation is unavoidable and a partner's tasting panels will speed it up. "
    "The main risk is loss of control over brand quality. We propose to hold 51% of the "
    "venture, keep the right to audit the factory quarterly, and reserve the final say over "
    "packaging and recipes. If the partner refuses these terms, a licensing deal with a "
    "smaller share of the profit is the fallback. Overall, a joint venture trades some "
    "margin for speed, and in a market where the first two years decide who wins shelf "
    "space, speed matters more than margin."
)

TEXT_A_PARAPHRASE = (
    "Rather than setting up a fully owned subsidiary, Northwind Foods ought to break into "
    "Vietnam's snack sector by partnering with a local firm. Three findings support this "
    "view. To begin with, getting products to shoppers there is difficult because the trade "
    "is scattered: about seven in ten snack purchases happen in tiny family stores which no "
    "national retailer serves. A domestic ally already holds those ties, whereas creating "
    "them ourselves would consume four or five years. Next, permission to bring in foreign "
    "flavour ingredients may take as long as eleven months, but a partner that already holds "
    "a permit shortens the wait to around three months. Finally, shoppers in that country "
    "favour sweet and salty tastes that our existing products lack, which makes adapting the "
    "recipes mandatory, and a partner's taste-testing groups would accelerate the work. The "
    "chief danger is that we lose command over the quality of the brand. We suggest owning "
    "51% of the joint company, keeping the entitlement to inspect the plant every quarter, "
    "and retaining the last word on recipes and packaging. Should the partner reject these "
    "conditions, we could fall back on licensing and accept a smaller profit share. In "
    "short, partnering sacrifices part of the margin in exchange for pace, and since the "
    "first two years determine who obtains shelf space, pace outweighs margin."
)

# Three pages; the paragraph read aloud in the MP4 pitch is on page 2.
SC_PAGES = [
    [
        "Executive overview",
        "Meridian Logistics closed the third quarter with revenue of 48.2 million dollars, up "
        "6% on the previous year. Volumes rose in every region except the Gulf, where a port "
        "strike removed nine working days. Management attributes the growth mainly to the new "
        "cold-chain contracts signed in April, which now account for a fifth of revenue.",
        "Headcount was flat at 1,140, and overtime fell for the second quarter running as the "
        "new scheduling software bedded in. This review covers operations, cost pressures and "
        "the outlook for the final quarter of the year.",
    ],
    [
        "Operations and cost pressures",
        "The Dallas warehouse automation project finished eleven days ahead of plan. Pick rates "
        "there have doubled since launch, though we are still training staff to handle the "
        "exceptions that the robots cannot.",
        "Operating margins contracted by 4.2% over Q3 due to supply chain bottlenecks at the "
        "Rotterdam hub. Fuel surcharges rose faster than we could pass them on to customers. "
        "Two long-haul carriers failed to meet agreed delivery windows, and we paid penalty "
        "fees on thirty-one shipments. In response we have moved forty percent of Rotterdam "
        "volume to the Antwerp terminal.",
        "We also renegotiated the carrier contracts to include stricter penalties. Early data "
        "from October suggests transit times have improved by almost a day.",
    ],
    [
        "Outlook",
        "For the fourth quarter we expect seasonal volumes to lift revenue by roughly nine "
        "percent. The Singapore expansion remains on schedule for February, subject to customs "
        "approval. We do not plan further price increases before the new fiscal year.",
        "The board will review the capital budget in December, with automation at two more "
        "sites the leading candidate for investment.",
    ],
]

# What the speaker says in 04_pitch_supply_chain.mp4 (reads page 2, with one slip)
VIDEO_SC_SENTENCES = [
    "Operating margins contracted by 4.2% over Q3 due to supply chain bottlenecks at the "
    "Rotterdam hub.",
    "Fuel surcharges rose faster than we could pass on to customers.",
    "Two long-haul carriers failed to meet agreed delivery windows, and we paid penalty fees "
    "on thirty-one shipments.",
    "In response we have moved forty percent of Rotterdam volume to the Antwerp terminal.",
]

VIDEO_ORIGINAL_SENTENCES = [
    "Hi everyone, we are building a bicycle sharing app for university campuses.",
    "Students unlock a bike with their phone, ride between lecture halls, and park it at any "
    "marked rack.",
    "Our pilot with two hundred bikes showed that each bike was used seven times a day.",
    "We want to expand to ten campuses by next autumn.",
]

AI_ESSAY = [
    "Artificial intelligence is transforming the landscape of modern education in profound and "
    "multifaceted ways. In today's rapidly evolving digital world, it is important to recognize "
    "that AI-powered tools are not only enhancing the learning experience but also reshaping how "
    "educators deliver content. Furthermore, personalized learning platforms leverage data "
    "analytics to tailor instruction to the unique needs of each student, thereby fostering "
    "greater engagement and improved outcomes.",
    "Moreover, AI plays a crucial role in streamlining administrative tasks. Automated grading "
    "systems, intelligent scheduling tools, and predictive analytics enable educators to focus on "
    "what truly matters: nurturing student growth. Additionally, virtual tutors provide students "
    "with seamless, round-the-clock support, ensuring that learners can receive assistance "
    "whenever they need it.",
    "However, it is essential to acknowledge the challenges associated with integrating AI into "
    "educational settings. Concerns regarding data privacy, algorithmic bias, and the digital "
    "divide underscore the importance of responsible implementation. Stakeholders must "
    "collaborate to establish robust ethical frameworks that safeguard student interests while "
    "harnessing the transformative potential of technology.",
    "In conclusion, artificial intelligence represents a pivotal opportunity to enhance education "
    "across the globe. By embracing a holistic approach that balances innovation with ethical "
    "considerations, institutions can navigate the complexities of this ever-evolving landscape "
    "and foster a brighter future for learners everywhere. Overall, the key takeaway is clear: "
    "the successful integration of AI in education requires thoughtful planning, comprehensive "
    "training, and a steadfast commitment to equity.",
]

CASE_PROMPT = (
    "GreenLoop Packaging is a mid-sized manufacturer of compostable food containers based in "
    "Pune, India. Over the last three years its revenue has grown by 18% annually, but gross "
    "margin has fallen from 34% to 27% because the price of its main input, sugarcane bagasse, "
    "has risen while large restaurant-chain customers have demanded lower prices. The CEO is "
    "deciding between three options: (a) build a second plant closer to the sugarcane belt, "
    "(b) sign a long-term supply contract with a single bagasse supplier, or (c) develop a "
    "premium product line for health-conscious retail consumers. Instructions: Your submission "
    "must (1) identify the core problem, (2) evaluate each option against cost, risk and speed, "
    "(3) recommend one option with a clear implementation plan, and (4) list the assumptions on "
    "which your recommendation depends. Do not exceed 1,000 words. Cite any outside data you use."
)

CASE_ANSWERS = {
    "alpha": (
        "The root problem is not demand; it is that GreenLoop buys its main input at a price it "
        "cannot control and sells to customers who can. Revenue is up 18% a year yet margin has "
        "dropped seven points, so every extra tray sold earns less than the last. Option (a), a "
        "second plant near the cane belt, attacks the cost directly. Bagasse is bulky and cheap "
        "per kilo, so trucking it 600 km from the sugar mills to Pune eats a large share of its "
        "value; a plant next to the mills could plausibly cut landed input cost by 12 to 15 "
        "percent in my estimate. The downside is capital: I assume roughly 90 crore rupees and an "
        "18-month build. Option (b) leaves GreenLoop dependent on one supplier, and (c) does "
        "nothing about the cost base. I therefore recommend (a), phased: lease a shed near "
        "Kolhapur in the first six months to prove the economics before committing to a full "
        "build. Key assumptions: mill owners will give long-term access to fresh bagasse, and the "
        "restaurant chains will not demand further price cuts during construction."
    ),
    "bravo": (
        "Let me start by questioning the framing. Is margin really the problem, or is it "
        "volatility? Bagasse prices moved around a lot, and that unpredictability hurts planning "
        "more than the level itself. That points me toward option (b). A five-year contract with "
        "one reliable supplier fixes the price band, which lets finance quote restaurant "
        "customers with confidence. I do worry about single-supplier risk, so the contract should "
        "include a second-source clause and penalties for missed deliveries. A new plant costs "
        "too much and takes too long; a premium line is a separate bet on consumer behaviour. If "
        "I were the CEO, I would open talks this month with the three largest mills, push for an "
        "index-linked price with a cap, and accept a slightly higher average price in exchange "
        "for stability. Assumptions: suppliers want guaranteed volume more than they want "
        "spot-market upside, and GreenLoop's current output of about 400 tonnes a month is large "
        "enough to negotiate with."
    ),
    "charlie": (
        "GreenLoop is stuck competing on price with customers who hold all the cards. The way out "
        "is to sell to someone who cares more about the product than the price. Option (c), the "
        "premium retail line, does that. Health-conscious shoppers already pay 30 to 40 percent "
        "more for 'clean' packaged food, and a compostable container with a visible bagasse "
        "texture fits the story. I would test it in three Mumbai and Bengaluru supermarket chains "
        "first, using 5,000 units, before spending on new tooling. The risks are slow adoption and "
        "the cost of retail marketing, which restaurants never required. Options (a) and (b) are "
        "defensive moves that fix cost without changing the dynamic with big customers. "
        "Recommendation: pursue (c) with a pilot budget of about 25 lakh rupees, and review at "
        "six months with a go/no-go on sales per store per week. Assumption: retail consumers "
        "care about compostability enough to pay a premium, which our survey of 200 shoppers "
        "hints at but does not prove."
    ),
    "delta": (
        "I would sequence the options instead of picking one. In the first year, sign the supply "
        "contract (b) because it is quick, cheap, and stabilises the numbers while we think. "
        "Revenue growth is strong, so the company has time but not a lot of cash. In year two, use "
        "the calmer margins to fund a premium range (c) aimed at retailers. Only if volumes keep "
        "climbing past 600 tonnes a month would I revisit a second plant, since a half-empty "
        "factory would hurt more than expensive bagasse ever did. The core problem, in my view, is "
        "that GreenLoop is a price-taker on both sides. Sequencing reduces exposure on both fronts "
        "without a large bet. The assumptions: the supplier will accept a two-year term, retail "
        "pilots can be financed from operating cash flow, and growth does not stall. The plan is "
        "deliberately modest; if the numbers surprise us, we can accelerate."
    ),
}

BEEKEEPING = (
    "Urban Hive Co. wants to put beehives on the flat roofs of office buildings and sell the "
    "honey under a local label. The idea sounds like a gimmick until you look at the unit "
    "economics. One hive yields about 25 kilograms in a good season, and rooftop honey in our "
    "city retails at roughly 1,400 rupees per kilogram because buyers like the story. A hive "
    "costs 9,000 rupees to set up, so payback is under a season if the colony survives. That "
    "last condition is the catch. Heat on bare concrete roofs can kill a colony in May, so "
    "every site needs shade netting and a water tray, which adds about 2,500 rupees. We also "
    "learned that building managers care less about honey than about liability; the first "
    "three pitches failed until we offered insurance cover and a sting-free zone around the "
    "stairwell. Our current plan is twelve rooftops by March, with a part-time beekeeper "
    "visiting each fortnight. The risk I worry about most is not the bees but the permits: two "
    "municipal wards still treat hives as livestock. If that is not resolved by January, we "
    "will move the pilot to private campuses where the rules are simpler and the answer is "
    "faster."
)

KB = {
    "vertical_integration.txt": (
        "Vertical integration is a strategy in which a company takes ownership of several "
        "stages of its production and distribution chain rather than buying those stages from "
        "independent firms. Backward integration moves a firm toward its raw materials and "
        "suppliers, while forward integration moves it toward its customers, for example by "
        "opening its own retail outlets. Firms pursue it to secure supply, capture the margin "
        "earned by intermediaries, and protect proprietary know-how. The costs are significant: "
        "large capital commitments, reduced flexibility when technology or demand shifts, and "
        "the managerial difficulty of running businesses with very different economics. "
        "Economists often explain the trade-off through transaction costs. When contracts are "
        "hard to write or enforce, owning the activity is cheaper than negotiating for it "
        "repeatedly. When markets are competitive and standards are clear, buying from "
        "specialists is usually the more efficient choice."
    ),
    "supply_chain_basics.txt": (
        "A supply chain is the network of organisations, people, information and resources "
        "involved in moving a product from raw material to the final customer. Core activities "
        "include sourcing, manufacturing, warehousing, transportation and returns handling. "
        "Performance is commonly judged on cost, delivery reliability, speed and inventory "
        "turnover. A frequent problem is the bullwhip effect, where small changes in consumer "
        "demand produce increasingly large swings in orders further up the chain. Companies "
        "reduce it by sharing sales data with suppliers, shortening lead times and ordering in "
        "smaller, more frequent batches. Disruptions such as port closures or supplier failures "
        "have pushed many firms to hold safety stock and to qualify second suppliers."
    ),
    "market_entry_modes.txt": (
        "Companies entering a foreign market choose among several modes. Exporting requires "
        "little investment but gives limited control over local sales. Licensing lets a local "
        "firm produce and sell under the brand in return for royalties, at the risk of "
        "leaking know-how. A joint venture shares ownership and risk with a local partner who "
        "contributes market knowledge, although disagreements over control are common. A wholly "
        "owned subsidiary, built from scratch or acquired, offers the greatest control and the "
        "greatest cost. The choice usually depends on the size of the opportunity, the "
        "regulatory environment, cultural distance and how quickly the firm needs to move."
    ),
    "circular_economy.txt": (
        "A circular economy aims to keep products and materials in use for as long as possible "
        "by designing out waste. Strategies include repair, refurbishment, remanufacturing and "
        "recycling, and replacing disposable items with reusable or compostable alternatives. "
        "Proponents argue that it reduces dependence on virgin resources and exposure to price "
        "swings in commodities. Critics note that recycling is energy-intensive, that "
        "composting requires proper collection infrastructure, and that claims of "
        "compostability are often hard for consumers to verify."
    ),
    "working_capital.txt": (
        "Working capital is the difference between a company's current assets and its current "
        "liabilities, and it measures the short-term funds available to run daily operations. "
        "The cash conversion cycle combines the days inventory is held, the days customers take "
        "to pay, and the days the company takes to pay suppliers. Shortening the cycle frees "
        "cash without any new financing. Fast-growing firms often run short of working capital "
        "even while profitable, because they must buy inventory and extend credit before "
        "customers pay."
    ),
}

COMMENTARY_14 = (
    "Crumb & Co. is a regional bakery chain with 14 stores, and this report asks whether it "
    "should buy its own flour mill. The theory is straightforward, but the numbers decide it. "
    "Flour is about 22% of our cost of goods, and a 10% rise last year cut store margin by "
    "nearly a point. A small mill would cost 3.5 crore rupees and needs a steady 40 tonnes "
    "of wheat a week to break even, which is more than our stores use today. That mismatch "
    "worries me. I would rather sign a two-year flour contract and revisit the mill once we "
    "pass 20 stores. I would also ask the owner whether the mill could sell surplus flour to "
    "local restaurants, because that single question changes the payback from nine years to "
    "five."
)
