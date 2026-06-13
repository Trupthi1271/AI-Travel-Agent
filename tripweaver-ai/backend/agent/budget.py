def calculate_budget(input_text: str) -> str:
    """
    Calculate daily budget breakdown from total budget and number of days.
    Input format: 'AMOUNT,DAYS' e.g. '15000,3'
    """
    try:
        parts = [p.strip() for p in input_text.split(",")]
        if len(parts) != 2:
            return (
                "❌ Invalid format. Please provide input as: AMOUNT,DAYS\n"
                "Example: 15000,3 (₹15,000 for 3 days)"
            )

        total_budget = float(parts[0].lower().replace("₹", "").replace("rs", "").replace("inr", "").strip())
        days = int(parts[1])

        if total_budget <= 0:
            return "❌ Budget must be greater than ₹0."
        if days <= 0:
            return "❌ Number of days must be at least 1."

        per_day = total_budget / days

        # Realistic breakdown percentages
        accommodation = per_day * 0.40
        food = per_day * 0.25
        transport = per_day * 0.20
        activities = per_day * 0.15

        # Budget tier classification
        if per_day < 800:
            tier = "🎒 Ultra Budget (backpacker)"
            tip = "Consider hostels, street food, and local buses to stay within budget."
        elif per_day < 2000:
            tier = "💚 Budget Traveler"
            tip = "Budget guesthouses and local dhabas will keep you comfortable."
        elif per_day < 5000:
            tier = "🏨 Mid-Range"
            tip = "3-star hotels and good restaurants are well within reach."
        elif per_day < 10000:
            tier = "✨ Comfort Traveler"
            tip = "4-star hotels and curated experiences are affordable."
        else:
            tier = "👑 Luxury"
            tip = "5-star resorts and premium experiences are within budget."

        return (
            f"## 💰 Budget Breakdown\n\n"
            f"**Total:** ₹{total_budget:,.0f} for {days} day{'s' if days > 1 else ''} · "
            f"**Daily:** ₹{per_day:,.0f}/day · **Tier:** {tier}\n\n"
            f"| Category | Per Day | Total ({days}d) |\n"
            f"|---|---|---|\n"
            f"| 🏨 Accommodation | ₹{accommodation:,.0f} | ₹{accommodation*days:,.0f} |\n"
            f"| 🍽️ Food | ₹{food:,.0f} | ₹{food*days:,.0f} |\n"
            f"| 🚌 Transport | ₹{transport:,.0f} | ₹{transport*days:,.0f} |\n"
            f"| 🎯 Activities | ₹{activities:,.0f} | ₹{activities*days:,.0f} |\n"
            f"| **Total** | **₹{per_day:,.0f}** | **₹{total_budget:,.0f}** |\n\n"
            f"💡 {tip}"
        )

    except ValueError:
        return (
            "❌ Could not parse budget. Please use the format: AMOUNT,DAYS\n"
            "Example: 15000,3"
        )
    except Exception as e:
        return f"❌ Budget calculation error: {str(e)}"
