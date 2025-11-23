import csv
import random

# Configuration
START_ID = 1201
NUM_PAIRS = 600
OUTPUT_FILE = 'legal_pairs_supplement.csv'

# ------------------------------------------------------------------
# vocabulary and Synonyms (To avoid repetition)
# ------------------------------------------------------------------

# Replacements for the "overused" phrases to create diversity
compliance_phrases = [
    "subject to all applicable laws and regulations", # Keep original occasionally
    "in compliance with pertinent statutes",
    "adhering to relevant legal requirements",
    "governed by valid enactments",
    "pursuant to governing ordinances",
    "contingent upon statutory mandates"
]

dispute_phrases = [
    "if any dispute arises out of or relates to this agreement", # Keep original occasionally
    "should a controversy emerge regarding this compact",
    "in the event of a legal conflict stemming from this deal",
    "regarding any claims originating from this contract",
    "for all disagreements connected to this bond"
]

exception_phrases = [
    "except as otherwise expressly provided in this agreement", # Keep original occasionally
    "unless distinctly stipulated elsewhere herein",
    "barring specific exclusions noted in this text",
    "save for particular instances detailed within",
    "unless stated otherwise in this document"
]

extent_phrases = [
    "to the fullest extent permitted by law", # Keep original occasionally
    "to the maximum range allowed by statute",
    "as far as the governing law permits",
    "to the widest scope authorized by legislation",
    "within the full bounds of the law"
]

unenforceable_phrases = [
    "if any provision of this agreement is held to be unreliable or unenforceable", # Keep original occasionally
    "should a court deem any clause invalid",
    "if a tribunal finds a section void",
    "in the event a term is ruled ineffective",
    "if any part of this text is judged legally void"
]

condition_phrases = [
    "as a condition precedent to the exercise of any rights under this agreement", # Keep original occasionally
    "before utilizing any privileges granted herein",
    "prior to the enforcement of any claims under this pact",
    "as a prerequisite to activating rights within this deal",
    "before any contractual powers are exercised"
]

# Legal Entities
parties_a = ["Lessor", "Licensor", "Employer", "Indemnitor", "Vendor", "Disclosing Party", "Plaintiff"]
parties_b = ["Lessee", "Licensee", "Employee", "Indemnitee", "Purchaser", "Receiving Party", "Defendant"]

# ------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------
# Each template has a 'legal' structure and a 'plain' structure.
# We use placeholders like {A}, {B}, {COMPLIANCE}, etc.

templates = [
    {
        "type": "Indemnification",
        "legal": "The {A} shall indemnify, defend, and hold harmless the {B} from any and all claims, damages, or liabilities arising from the {A}'s negligence, {COMPLIANCE}.",
        "plain": "The {A_PLAIN} must pay for any losses or legal issues the {B_PLAIN} faces because of the {A_PLAIN}'s carelessness, as long as it follows the law."
    },
    {
        "type": "Termination",
        "legal": "This Agreement may be terminated forthwith by the {B} {EXCEPTION}, provided that written notice is dispatched no less than thirty days prior to the effective date.",
        "plain": "The {B_PLAIN} can end this contract immediately, unless the contract says otherwise, by sending a written warning at least 30 days in advance."
    },
    {
        "type": "Confidentiality",
        "legal": "The {B} covenants to maintain the secrecy of the Proprietary Information and shall not disclose said data to third parties {EXTENT}.",
        "plain": "The {B_PLAIN} promises to keep the private info secret and not share it with outsiders as much as the law allows."
    },
    {
        "type": "Severability",
        "legal": "{UNENFORCEABLE}, the remaining portions shall remain in full force and effect, preserving the original intent of the parties.",
        "plain": "If a court decides one part of this contract is broken, the rest of the contract stays valid and keeps the original meaning."
    },
    {
        "type": "Jurisdiction",
        "legal": "{DISPUTE}, exclusive jurisdiction and venue shall lie within the courts of the State of Delaware.",
        "plain": "If there is a fight about this deal, the courts in Delaware will be the only place to handle it."
    },
    {
        "type": "Waiver",
        "legal": "The failure of the {A} to enforce any right hereunder shall not constitute a waiver of such right, nor shall it preclude subsequent enforcement.",
        "plain": "Just because the {A_PLAIN} doesn't enforce a rule right away doesn't mean they lose the right to enforce it later."
    },
    {
        "type": "Force Majeure",
        "legal": "Neither party shall be liable for failure to perform obligations due to causes beyond their reasonable control, including acts of God, {COMPLIANCE}.",
        "plain": "Neither side is responsible for missing work due to major disasters they can't control, following relevant laws."
    },
    {
        "type": "Assignment",
        "legal": "This Agreement and the rights herein may not be assigned by the {B} without the prior written consent of the {A}, {EXCEPTION}.",
        "plain": "The {B_PLAIN} cannot transfer this contract to anyone else without the {A_PLAIN}'s written permission, unless this deal says otherwise."
    },
    {
        "type": "Entire Agreement",
        "legal": "This instrument embodies the entire agreement between the parties and supersedes all prior negotiations, understandings, and representations.",
        "plain": "This document contains the whole deal and replaces any previous talks or promises."
    },
    {
        "type": "Notices",
        "legal": "All notices required hereunder shall be in writing and deemed effective upon receipt when sent via certified mail, return receipt requested.",
        "plain": "All official alerts must be written and count as delivered when they arrive via certified mail."
    },
    {
        "type": "Intellectual Property",
        "legal": "All intellectual property rights developed during the term of this Agreement shall vest exclusively in the {A}, {EXTENT}.",
        "plain": "The {A_PLAIN} will own all creative rights made during this contract to the full limit of the law."
    },
    {
        "type": "Payment",
        "legal": "Payment shall be remitted by the {B} within thirty (30) days of the invoice date, and late payments shall incur interest at a rate of 1.5% per month.",
        "plain": "The {B_PLAIN} must pay within 30 days of the bill, and late payments will include a 1.5% monthly interest fee."
    },
    {
        "type": "Audit Rights",
        "legal": "{CONDITION}, the {A} shall have the right to inspect and audit the books and records of the {B} during normal business hours.",
        "plain": "Before using rights in this deal, the {A_PLAIN} can check the {B_PLAIN}'s records during standard work hours."
    },
    {
        "type": "Modification",
        "legal": "No amendment, modification, or waiver of any provision of this Agreement shall be effective unless in writing and signed by both parties.",
        "plain": "Changes to this contract are only valid if they are written down and signed by both sides."
    },
    {
        "type": "Warranty Disclaimer",
        "legal": "The {A} makes no representations or warranties, express or implied, regarding the fitness for a particular purpose of the goods.",
        "plain": "The {A_PLAIN} does not promise that the goods will work for any specific job."
    },
    {
        "type": "Relationship of Parties",
        "legal": "Nothing contained herein shall be construed to create a partnership, joint venture, or agency relationship between the {A} and the {B}.",
        "plain": "This contract does not make the {A_PLAIN} and {B_PLAIN} partners or agents of each other."
    },
    {
        "type": "Counterparts",
        "legal": "This Agreement may be executed in counterparts, each of which shall be deemed an original, but all of which together shall constitute one and the same instrument.",
        "plain": "This contract can be signed in separate copies, but they all count as one single agreement."
    },
    {
        "type": "Survival",
        "legal": "The obligations concerning confidentiality and indemnification shall survive the expiration or termination of this Agreement for a period of five years.",
        "plain": "Duties regarding secrecy and paying for damages will last for five years after this contract ends."
    },
    {
        "type": "Time of Essence",
        "legal": "Time is of the essence with respect to all obligations and performance dates specified in this Agreement.",
        "plain": "Deadlines are very important for all tasks and dates in this deal."
    },
    {
        "type": "Return of Property",
        "legal": "Upon termination, the {B} shall explicitly return all materials, equipment, and documents belonging to the {A} forthwith.",
        "plain": "When the deal ends, the {B_PLAIN} must immediately give back all of the {A_PLAIN}'s stuff and papers."
    },
    {
        "type": "Taxes",
        "legal": "The {B} shall be responsible for all applicable federal, state, and local taxes arising from the transactions contemplated hereby, {COMPLIANCE}.",
        "plain": "The {B_PLAIN} must pay all taxes related to this deal, obeying relevant laws."
    },
    {
        "type": "Non-Solicitation",
        "legal": "The {B} agrees not to solicit, recruit, or hire any employee of the {A} for a period of twelve months following termination.",
        "plain": "The {B_PLAIN} agrees not to try to hire the {A_PLAIN}'s workers for one year after the contract ends."
    },
    {
        "type": "Insurance",
        "legal": "The {B} shall maintain comprehensive general liability insurance with limits of not less than one million dollars per occurrence.",
        "plain": "The {B_PLAIN} must keep liability insurance covering at least one million dollars per incident."
    },
    {
        "type": "Subcontracting",
        "legal": "The {B} may not subcontract any portion of the services to be performed hereunder without the {A}'s prior written approval.",
        "plain": "The {B_PLAIN} cannot hire others to do the work without the {A_PLAIN}'s written okay."
    },
    {
        "type": "Exclusivity",
        "legal": "During the Term, the {A} shall be the sole and exclusive provider of the Services to the {B} within the Territory.",
        "plain": "While this contract lasts, only the {A_PLAIN} can provide these services to the {B_PLAIN} in this area."
    },
    {
        "type": "Publicity",
        "legal": "Neither party shall issue any press release or public announcement relating to this Agreement without the other party's prior written consent.",
        "plain": "Neither side can make public statements about this deal without the other's written permission."
    },
    {
        "type": "Remedies Cumulative",
        "legal": "The rights and remedies provided in this Agreement are cumulative and not exclusive of any rights or remedies provided by law.",
        "plain": "The rights in this deal are added to any rights given by law, not replacing them."
    },
    {
        "type": "Headings",
        "legal": "The headings and captions in this Agreement are for convenience only and shall not affect the interpretation or construction of this Agreement.",
        "plain": "The titles in this document are just for easy reading and don't change the contract's meaning."
    },
    {
        "type": "Further Assurances",
        "legal": "Each party shall execute and deliver such further instruments as may be reasonably necessary to carry out the intent of this Agreement.",
        "plain": "Each side will sign any extra papers needed to make this agreement work."
    },
    {
        "type": "Successors",
        "legal": "This Agreement shall be binding upon and inure to the benefit of the parties hereto and their respective successors and assigns.",
        "plain": "This contract applies to the parties and anyone who takes over their business or rights."
    }
]

# ------------------------------------------------------------------
# Generator Logic
# ------------------------------------------------------------------

def get_weighted_choice(options):
    """
    Selects a phrase. 
    Logic: The original 'overused' phrase (usually index 0) is weighted lower 
    to prevent overfitting, but still present.
    """
    if not options: return ""
    # 20% chance for the 'standard/overused' phrase, 80% split among variations
    weights = [0.2] + [0.8 / (len(options)-1)] * (len(options)-1)
    return random.choices(options, weights=weights, k=1)[0]

def generate_row(row_id):
    template = random.choice(templates)
    
    # Pick random entities
    # To ensure consistency within the row (e.g. if A is Lessor, B should be Lessee)
    # We map indices. 
    idx = random.randint(0, len(parties_a) - 1)
    a_legal = parties_a[idx]
    b_legal = parties_b[idx]
    
    # Plain versions usually correspond, or are generic
    # Simple mapping for plain English
    plain_map = {
        "Lessor": "landlord", "Lessee": "tenant",
        "Licensor": "licensor", "Licensee": "licensee",
        "Employer": "employer", "Employee": "worker",
        "Indemnitor": "payer", "Indemnitee": "protected party",
        "Vendor": "seller", "Purchaser": "buyer",
        "Disclosing Party": "sharer", "Receiving Party": "receiver",
        "Plaintiff": "plaintiff", "Defendant": "defendant"
    }
    
    a_plain = plain_map.get(a_legal, "party A")
    b_plain = plain_map.get(b_legal, "party B")

    # Select phrases
    compliance = get_weighted_choice(compliance_phrases)
    dispute = get_weighted_choice(dispute_phrases)
    exception = get_weighted_choice(exception_phrases)
    extent = get_weighted_choice(extent_phrases)
    unenforceable = get_weighted_choice(unenforceable_phrases)
    condition = get_weighted_choice(condition_phrases)

    # Fill Legal Template
    legal_text = template["legal"].format(
        A=a_legal, 
        B=b_legal, 
        COMPLIANCE=compliance,
        DISPUTE=dispute,
        EXCEPTION=exception,
        EXTENT=extent,
        UNENFORCEABLE=unenforceable,
        CONDITION=condition
    )

    # Fill Plain Template
    # Note: Plain templates are written to be direct, they don't necessarily 
    # need the complex variable substitutions for the "jargon" phrases 
    # because the plain translation simplifies those away naturally.
    plain_text = template["plain"].format(
        A_PLAIN=a_plain,
        B_PLAIN=b_plain
    )

    return [row_id, legal_text, plain_text]

def main():
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'src_legal', 'tgt_plain'])
        
        for i in range(NUM_PAIRS):
            row_id = START_ID + i
            row = generate_row(row_id)
            writer.writerow(row)
            
    print(f"Successfully generated {NUM_PAIRS} pairs in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()