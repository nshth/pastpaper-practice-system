from PyPDF2 import PdfReader, PdfWriter

# Load PDF
pdf_path = input("Enter PDF file path: ")
reader = PdfReader(pdf_path)

total_pages = len(reader.pages)
print(f"Total pages: {total_pages}")

# Ask where to split
split_index = int(input(f"Enter split index (0 to {total_pages}): "))

# Validate
if split_index < 0 or split_index > total_pages:
    print("Invalid split index.")
    exit()

# Split into two parts
part1 = reader.pages[:split_index]
part2 = reader.pages[split_index:]

print(f"Part 0 = pages 0 to {split_index-1}")
print(f"Part 1 = pages {split_index} to {total_pages-1}")

# Ask which part to keep
choice = int(input("Which part to save? (0 or 1): "))

writer = PdfWriter()

if choice == 0:
    for page in part1:
        writer.add_page(page)
elif choice == 1:
    for page in part2:
        writer.add_page(page)
else:
    print("Invalid choice.")
    exit()

# Save output
output_path = input("Enter output file name (press Enter to replace original): ").strip()

# If empty → overwrite original
if output_path == "":
    output_path = pdf_path
    print("No name given. Replacing original file...")

with open(output_path, "wb") as f:
    writer.write(f)

print("Done.")