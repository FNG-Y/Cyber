package game;

public class Book {
    private final String title;
    private final String author;
    private final int pages;
    private final Skill skill;

    public Book(String title, String author, int pages, Skill skill) {
        this.title = title;
        this.author = author;
        this.pages = pages;
        this.skill = skill;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public int getPages() {
        return pages;
    }

    public Skill getSkill() {
        return skill;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Book book = (Book) o;
        return title.equals(book.title) && author.equals(book.author);
    }
}