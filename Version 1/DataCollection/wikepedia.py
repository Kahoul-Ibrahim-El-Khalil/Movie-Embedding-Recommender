import wikipediaapi
import wikitextparser as wtp

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MovieDataCollectorBot/1.0 (kahoul.ibraim.elkhalil@gmail.com)'
)

def get_movie_release_date(Page):
    try:
        raw_text = Page.content
        parsed = wtp.parse(raw_text)
        infoboxes = parsed.templates
        for box in infoboxes:
            if 'film' in box.name.lower():
                release = box.get_arg("released") or box.get_arg("release_date")
                if release:
                    return release.value.strip()
        return "NaN"
    except Exception:
        return "NaN"

def get_movie_data_of_wikipedia(Title):
    page = wiki.page(Title)
    if page.exists():
        plot_section = page.section_by_title("Plot")
        if plot_section:
            description = plot_section.text
        else:
            description = page.summary
        return {
            "Title":  page.title,
            "Release Date":  get_movie_release_date(page),
            "Description":  description
        }
    else:
        return None

def get_data_of_wikipedia(Titles):
    data = []
    failures = []
    for title in Titles:
        info = get_movie_data_of_wikipedia(title)
        if info:
            data.append(info)
        else:
            failures.append(title)
    return (data, failures)

