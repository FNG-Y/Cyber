package game;

import java.util.ArrayList;
import java.util.List;

public class Library {
    private final List<Book> books = new ArrayList<>();

    public void addBook(Book book) {
        books.add(book);
    }

    public boolean removeBook(Book book) {
        return books.remove(book);
    }

    public boolean assignBestSkillBook(Hero hero, Skill desiredSkill) {
        if (hero.getCurrentSkill() == desiredSkill) {
            return false;
        }

        Book bestBook = null;
        for (Book book : books) {
            if (book.getSkill() == desiredSkill) {
                if (bestBook == null || book.getPages() < bestBook.getPages()) {
                    bestBook = book;
                }
            }
        }

        if (bestBook == null) {
            return false;
        }

        hero.readBook(bestBook);
        removeBook(bestBook);
        return true;
    }
}