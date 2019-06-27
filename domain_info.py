# DOMAIN INFORMATION ----------------------------------------------------------
# 
# Information for domains in different datasets for experiments. List of stopwords also.


# AMAZON INFORMATION ----------------------------------------------------------

domain_names = ["clothing", "camera", "cuisine", "magazine", "software", "toy", 
                "car", "cell phone", "grocery/food", "music", "sports", "video",
                "baby", "video_games", "health/healthy lifestyle", "cosmetics",
                "movie", "jewellry/watches", "book", "electronics", "kitchen/house", 
                "patio/garden"]

a_d0 = """Clothing is an item or fabric, usually sewn together to cover part of the human body. 
Humans are the only animals which wear clothing, and all people do wear suitable clothing. 
The torso (body) can be covered by shirts, arms by sleeves, legs by pants or skirts, hands 
by gloves, feet by footwear, and head by headgear or masks. In cold climates, people also 
wear heavy, thick coats such as trenchcoats. Clothing protects the human body from the hot sun 
and high temperatures in warm tropical countries. Clothing such as thick wool coats and boots 
keeps the human body warm in very cold temperatures (such as in the arctic). To some extent, 
clothing protects people from damage to their body. Clothing is also worn for decoration, 
as a fashion (clothing). People from different cultures wear different clothing, and have 
different beliefs and customs about what type of clothing should be worn. For many people, 
clothing is a status symbol. It helps people project an image. Often, clothing is a form of 
self-expression. Adults in different social or work situations present different views of 
themselves by the clothes they wear. Young people have an entirely different form of dress 
to express their personalities. Often people will simply follow popular fashion styles so that 
they will fit in. Clothing is far more than just a means to protect our bodies."""

a_d1 = """A camera is a device that takes pictures (photographs). It uses film or electronics to 
make a picture of something. It is a tool of photography. A lens makes the image that the 
film or electronics "sees". A camera that takes one picture at a time is sometimes called 
a still camera. A camera that can take pictures that seem to move is called a movie camera. 
If it can take videos it is called a video camera or a camcorder. The majority of cameras 
are on a phone. This is called a "Camera phone". All cameras are basically a box that light 
can not get into until a photo is taken. There is a hole on one side of the camera where the 
light can get in through the lens, and this is called the aperture. On the other side is a 
special material that can record the image that comes through the aperture. This material is 
the film in a film camera or electronic sensor in a digital camera. Finally, there is also 
the shutter, which stops light from getting in until a photo is taken. When a photo is taken, 
the shutter moves out of the way. This lets light come in through the aperture and make a 
picture on the film or electronic sensor. In many cameras, the size of the aperture can be 
changed to let in more light or less light. The amount of time that the shutter lets light 
through can be changed as well. This also lets in more light or less light. Most of the time, 
electronics inside the camera control these, but in some cameras the person taking the picture 
can change them as well."""

a_d2 = """Cuisine refers to any style of cooking, including its practices, traditions and recipes.
A cuisine is usually associated with a specific culture. It is mainly influenced by the 
ingredients that are available to that culture. Cooking methods, customs and ingredients 
together form meals that are unique to a particular region. When people talk about Italian 
cuisine, they are referring to the food that Italians are famous for. Cuisine is a French 
word that means "kitchen", but it originally comes from the Latin word coquere, which means 
"to cook". Traditionally, cuisines are shaped by a number of things. In some religious traditions, 
certain foods are forbidden by law (an example is in Judaism). Climate and trade affect the what 
ingredients are available. Climate may also affect the methods by which food is prepared. Before 
the exploration of Asia and America by Europeans, certain foods were not known to European 
cuisine. Tomatoes, maize, avocados and cacao, for example, did not enter European diets until 
merchants brought these ingredients back from the New World. In modern times, cuisines are 
becoming more and more multicultural. New cuisines continue to develop even today."""

a_d3 = """A magazine is a type of book people read. Magazines are not like regular books. This 
is because a new version of the magazine is printed many times each year. Magazines are a 
type of periodical. They are called periodicals because new versions of them keep being printed. 
Magazines are printed on paper. People usually need to subscribe to them. An example of a magazine 
is Time. There are magazines printed about many things. Magazines are similar to newspapers, 
but usually new versions take longer to make, they cost more money, and they are in color on 
every page. Also, sometimes magazines come with little gifts to reward the readers who buy it."""

a_d4 = """Computer software, often called as software, is a set of instructions and its associated 
documentations that tells a computer what to do or how to perform a task. Software includes all 
different software programs on a computer, such as applications and the operating system. 
Applications are programs that are designed to perform a specific operation, such as a game 
or a word processor. The operating system including Mac OS, Microsoft Windows and Linux is 
an open source software that is used as a platform for running the applications, and controls 
all user interface tools including display and the keyboard. People, processes and tools are 
considered as essential ingredient of software design. The word software was first used in the 
late 1960s to emphasize on its difference from computer hardware, which can be physically observed 
by the user. Software is a set of instructions that the computer follows. Before compact discs 
(CDs) or development of the Internet age, software was used on various computer data storage 
media tools like paper punch cards, magnetic discs or magnetic tapes. The word firmware is 
sometimes used to describe a style of software that is made specifically for a particular type 
of computer or an electronic device and is usually stored on a Flash memory or ROM chip in the 
computer. Firmware usually refers to a piece of software that directly controls a piece of 
hardware. The firmware for a CD drive or the firmware for a modem are examples of firmware 
implementation. Today, software has become an inseparable part of our lives. We are using 
software everywhere. Software engineers are responsible for producing fault-free software 
which has literally become an essential part of our daily lives. Changeability and conformity 
are two of the main properties of software design. There are also different processing models 
for designing software including Build and Fix, Waterfall and Agile software processing design methods."""

a_d5 = """A toy is something to play with. Toys are for children, adults, and animals. Before 1970, 
most toys were made of metal and wood. Now, they are mostly made of plastic. Sometimes they are 
made of electronic material. Some people also consider video games toys. Toys include balls, 
plastic cars, and dolls. They also can be sex related products. Toys originated many years ago. 
Dolls of infants, animals, and soldiers and toy tools are found in archaeological places. 
The origin of the word "toy" is not known. However, it is thought that it was first used in 
the 14th century. The earliest toys were made from rocks, sticks or clay. In Ancient Egypt, 
children played with dolls made of clay and sticks.[2] In Ancient Greece and Ancient Rome, 
children played with dolls made of wax or terracotta, sticks, bows and arrows, and yo-yos. Later 
these developmental games were used in creation of indoor playsets for homes [3]. When Greek 
children, especially girls became adults, it was required to sacrifice the toys to the gods. 
On the eve of their wedding, young girls around fourteen would offer their dolls in a temple as 
a rite of passage into adulthood."""

a_d6 = """A car is a road vehicle used to carry passengers. It is also called an automobile, which 
comes from the Greek word "auto" and the French word "mobile". This name means "self-moving", as 
cars do not need horses or other external sources of power to move. Cars usually have four wheels 
and get their power from an engine. Millions of cars were made during the 20th century."""

a_d7 = """A mobile phone (also known as a hand phone, cell phone, or cellular telephone) is a 
small portable radio telephone. The mobile phone can be used to communicate over long distances 
without wires. It works by communicating with a nearby base station (also called a "cell site") 
which connects it to the main phone network. When moving, if the mobile phone gets too far away 
from the cell it is connected to, that cell sends a message to another cell to tell the new cell 
to take over the call. This is called a "hand off," and the call continues with the new cell the 
phone is connected to. The hand-off is done so well and carefully that the user will usually never 
even know that the call was transferred to another cell. As mobile phones became more popular, 
they began to cost less money, and more people could afford them. Monthly plans became available 
for rates as low as US$30 or US$40 a month. Cell phones have become so cheap to own that they 
have mostly replaced pay phones and phone booths except for urban areas with many people. In the 
21st century, a new type of mobile phone, called smartphones, have become popular. Now, more 
people are using smartphones than the old kind of mobile phone, which are called feature phones."""

a_d8 = """Food is what people and animals eat to survive. Food usually comes from animals or plants. 
It is eaten by living things to provide energy and nutrition.[1] Food contains the nutrition that 
people and animals need to be healthy. The consumption of food is enjoyable to humans. It contains 
protein, fat, carbohydrates, vitamins, and minerals. Liquids used for energy and nutrition are 
often called "drinks". If someone cannot afford food they go hungry. Food for humans is mostly 
made through farming or gardening. It includes animal and vegetable sources. Some people refuse 
to eat food from animal origin, like meat, eggs, and products with milk in them. Not eating 
meat is called vegetarianism. Not eating or using any animal products is called veganism. 
A grocery store (or just grocery) is a retail store that sells food. Most grocery stores sell 
fresh fruit, vegetables, seafood, meat, and other prepared foods. The person who controls a 
grocery store is called a grocer. A grocer will order food from farmers or other people who 
send out farmers food to other grocery stores and restaurants."""

a_d9 = """Music is a form of art; an expression of emotions through harmonic frequencies. Music is 
also a form of entertainment that puts sounds together in a way that people like, find interesting 
or dance to. Most music includes people singing with their voices or playing musical instruments, 
such as the piano, guitar, drums or violin. The word music comes from the Greek word (mousike), 
which means "(art) of the Muses". In Ancient Greece the Muses included the goddesses of music, 
poetry, art, and dance. Someone who makes music is known as a musician."""

a_d10 = """A sport is commonly defined as an athletic activity that involves a degree of competition, 
such as tennis or basketball. Some games and many kinds of racing are called sports. A professional
at a sport is called an athlete. Many people play sports with their friends. They need coaches 
to teach or train teams or individuals how to do better. Sports can be played indoors or
outdoors and by individuals or teams. Some people like to watch other people play sports. 
Those who watch others playing sports are called fans. While some fans watch sports on television, 
others actually go to stadiums or other places where people pay to watch them in person. These 
fans are called spectators. People engage in many kinds of sports"""

a_d11 = """Video is a technology. It records moving images onto some medium. The recorder may be a 
separate machine such as a videocassette recorder (also called a VCR) or built into something else 
such as a video camera. A popular 20th century videotape format was VHS. It was used by many 
people to record television programmes onto cassettes. This is usually an analog format. In the 
21st century digital recording is used more often than analog recording. By extension, a video 
clip is a short movie. A music video usually stars someone who recorded an (audio) album. One 
purpose is to promote the album."""

a_d12 = """A baby is a general word used to describe a human until about age 1 or 2 years old. Other 
terms can be used to describe the baby's stage of development. These terms do not follow clear 
rules and are used differently by different people, and in different regions. For example, infant 
may be used until the baby can walk, while some use "infant" until the baby is one year old. From 
birth until to 3 months of age, a baby can be called a newborn. Depending on how many weeks 
gestation at birth, newborns may be termed premature, post-mature, or full term. The word infant 
comes from Latin: infans means "unable to speak". At birth, many parts of the newborn's skull 
are yet converted to bone, leaving "soft spots". Later in the child's life, these bones fuse 
together naturally. A protein called noggin is responsible for the delay in an infant's skull 
fusion. During labour and birth, the infant's skull changes shape as it goes through the birth canal. 
Sometimes this causes the child to be born with a misshapen or elongated head. It will usually 
return to normal in a short time. Some religions have ceremonies that occur after a new baby is 
born, such as Baptism, which involves party or fully covering the baby in water. After learning to 
walk, babies are often called toddlers, usually between one and three years of age. Sometimes,
a woman's baby may die before or during its birth. Babies which have died in this way (no matter 
what gender) are called stillborn babies, or miscarried babies."""

a_d13 = """Video games are electronic games played on a video screen (normally a television, a built-in 
screen when played on a handheld machine, or a computer). There are many types, or genres, of 
these games: role-playing games; shooters, first-person shooters, side-scrollers, and platformers 
are just a few. Some games include a clan chat like Clash of Clans and Clash Royale created by 
Supercell or Roblox made by David Baszucki and Minecraft by Mojang Video games usually come on CDs, 
DVDs or digital download. Many games used to come on cartridges. A specialised device used to 
play a video game at home is called a console. There have been many types of consoles and home 
computers used to play video games. Some of the first were the Atari 2600, the Sega Master System 
and Nintendo Entertainment System in the 1980s. Newer video game consoles are the Xbox One, 
PlayStation 4 and Nintendo Switch. The best selling video game console of all time is the 
PlayStation 2, made by Sony. People can also use computers to play games, which are sometimes 
called PC games. The older consoles do not have new games developed for them often, although 
console games are emulated for PCs (see emulator). This means that new computers can play many 
old console games along with games made just for new computers. Older games are often more popular 
emulated than when they were first on sale, because of the ease of download. People can play 
portable video games anywhere. Mobile devices (running operating systems such as iOS or Android) 
also can download games, making them portable game machines. Mobile phones have many games, 
some of them using a mobile emulator for games from consoles. Not all PC or console Games are 
on mobile or iPad/ iPod/Tablet. Competitions of video game players are called electronic sports 
games like Fifa !8 and Fifa Mobile."""

a_d14 = """Health is "a state of complete physical, mental, and social well-being and not merely the 
absence of disease" according to the World Health Organization (WHO). [1][2] Physical is about the 
body. Mental is about how people think and feel. Social talks about how people live with other people. 
It is about family, work, school, and friends. A healthy lifestyle is one which helps to keep 
and improve people's health and well-being.[1] Many governments and non-governmental 
organizations work at promoting healthy lifestyles. [2]  They measure the benefits with 
critical health numbers, including weight, blood sugar, blood pressure, and blood cholesterol. 
Healthy living is a lifelong effect. The ways to being healthy include healthy eating, 
physical activities, weight management, and stress management. Good health allows people 
to do many things."""

a_d15 = """Cosmetics (also called makeup, make up, or make-up) are products used to make the human 
body look different. Often cosmetics are used to make someone more attractive to one person, or 
to a culture or sub-culture. In Western culture, women are the main users of cosmetics. Their 
use by men is less frequent, except on stage, television and movies. Cosmetics are widely used 
in the world of acting. All cosmetics are temporary. They need to be renewed after a certain time. 
Cosmetics include lipstick, powders (e.g. blush, eyeshadow), and lotions as well as other things."""

a_d16 = """Movies, also known as films, are a type of visual communication which uses moving pictures 
and sound to tell stories or teach people something. People in every part of the world watch 
movies as a type of entertainment, a way to have fun. For some people, fun movies can mean 
movies that make them laugh, while for others it can mean movies that make them cry, or feel 
afraid. Most movies are made so that they can be shown on big screens at movie theatres and at home. 
After movies are shown on movie screens for a period of weeks or months, they may be marketed 
through several other media. They are shown on pay television or cable television, and sold or 
rented on DVD disks or videocassette tapes, so that people can watch the movies at home. You 
can also download or stream movies. Older movies are shown on television broadcasting stations.
A movie camera or video camera takes pictures very quickly, usually at 24 or 25 pictures (frames) 
every second. When a movie projector, a computer, or a television shows the pictures at that rate, 
it looks like the things shown in the set of pictures are really moving. Sound is either recorded 
at the same time, or added later. The sounds in a movie usually include the sounds of people 
talking (which is called dialogue), music (which is called the "soundtrack"), and sound effects, 
the sounds of activities that are happening in the movie (such as doors opening or guns being 
fired). In the 20th century the camera used photographic film. The product is still often called 
a "film" even though there usually is no film."""

a_d17 = """Jewellery (or jewelry) refers to any clothing accessory that is worn as a decoration.
In itself, jewellery has no other purpose than to look attractive. However, it is often added 
to items that have a practical function. Items such as belts and handbags are mainly 
accessories but may be decorated. A broach, often worn to keep a cloak closed, could be 
highly decorated with jewels. Necklaces, finger rings and earrings are the most usual 
kinds of jewellery. A watch is a small clock carried or worn by a person. It makes it easy to 
see the time. It is also a fashion accessory for men and women, and expensive watches are 
designed for this purpose. A watch may be one of the few accessories worn by a man. A 
wristwatch is designed to be worn on a wrist, attached by a strap or other type of bracelet. 
A pocket watch is to be carried in a pocket. There are other variations. Nurses often wear 
a watch attached to the front of their uniform. It hangs down from a short strap: when 
lifted by the user it is right side up."""

a_d18 = """A book is a set of printed sheets of paper held together between two covers. 
The sheets of paper are usually covered with a text, language and illustrations that 
is the main point of a printed book. A writer of a book is called an author. Someone who 
draws pictures in a book is called an illustrator. Books can have more than one author 
or illustrator. A book can also be a text in a larger collection of texts. That way a 
book is perhaps written by one author, or it only treats one subject area. Books in 
this sense can often be understood without knowing the whole collection. Examples are the 
Bible, the Illiad or the Odyssey – all of them consist of a number of books in this sense 
of the word. Encyclopedias often have separate articles written by different people, and 
published as separate volumes. Each volume is a book. Hardcover books have hard covers 
made of cardboard covered in cloth or leather and are usually sewn together. Paperback 
books have covers of stiff paper and are usually glued together. The words in books can 
be read aloud and recorded on tapes or compact discs. These are called "audiobooks". Books 
can be borrowed from a library or bought from a bookstore. People can make their own books 
and fill them with family photos, drawings, or their own writing. Some books are empty 
inside, like a diary, address book, or photo album. Most of the time, the word "book" 
means that the pages inside have words printed or written on them. Some books are written 
just for children, or for entertainment, while other books are for studying something in 
school such as math or history. Many books have photographs or drawings."""

a_d19 = """Electronics is the study of how to control the flow of electrons. It deals with 
circuits made up of components that control the flow of electricity. Electronics is a 
part of physics and electrical engineering. Electrical components like transistors 
and relays can act as switches. This lets us use electrical circuits to process information 
and transmit information across long distances. Circuits can also take a weak signal 
(like a whisper) and amplify it (make it louder). Most electronic systems fall into 
two categories: Processing and distribution of information. These are called 
communications systems. Conversion and distribution of energy. These are called 
control systems. One way of looking at an electronic system is to separate it into 
three parts: Inputs - Electrical or mechanical sensors, which take signals from the 
physical world (in the form of temperature, pressure, etc.) and convert them into 
electric current and voltage signals. Signal processing circuits - These consist of 
electronic components connected together to manipulate, interpret and transform the 
information contained in the signals. Outputs - Actuators or other devices that transform 
current and voltage signals back into human readable information. A television set, 
for example, has as its input a broadcast signal received from an antenna, or for 
cable television, a cable. Signal processing circuits inside the television set 
use the brightness, colour, and sound information contained in the received signal 
to control the television set's output devices. The display output device may be a 
cathode ray tube (CRT) or a plasma or liquid crystal display screen. The audio 
output device might be a magnetically driven audio speaker. The display output 
devices convert the signal processing circuits' brightness and colour information 
into the visible image displayed on a screen. The audio output device converts 
the processed sound information into sounds that can be heard by listeners. Analysis of 
a circuit/network involves knowing the input and the signal processing circuit, and 
finding out the output. Knowing the input and output and finding out or designing 
the signal processing part is called synthesis."""

a_d20 = """A house is a building that is made for people to live in. It is a "permanent" 
building that is meant to stay standing. It is not easily packed up and carried away 
like a tent, or moved like a caravan. If people live in the same house for more than 
a short stay, then they call it their "home". Being without a home is called homelessness.
Houses come in many different shapes and sizes. They may be as small as just one room, 
or they may have hundreds of rooms. They also are made many different shapes, and may 
have just one level or several different levels. A house is sometimes joined to other 
houses at the sides to make a "terrace" or "row house" (a connected row of houses).
A big building with many levels and apartments is called "a block of flats" (British) 
or an apartment building. One of the differences between a house and an apartment is 
that a house has a front door to the outside world, whereas the main door of an apartment 
usually opens onto a passage or landing that can be used by other people in the building.
Houses have a roof to keep off the rain and sun, and walls to keep out the wind and cold. 
They have window openings to let in light, and a floor. Houses of different countries 
look different to each other, because of different materials, climate, and styles. The 
kitchen is the room in the house where food is cooked. Sometimes, people eat in their 
kitchens, too. Hotels, schools, and places where people work often have kitchens as 
well. A person who works in a kitchen in one of these places may be a kitchen worker 
or a chef (depending on where he/she cooks)."""

a_d21 = """A patio ("courtyard", "forecourt", "yard") is an outdoor space generally used 
for dining or recreation that adjoins a residence and is typically paved. In Australia 
the term is expanded to include roofed structures similar to a pergola, which provides 
protection from sun and rain. A garden is usually a piece of land that is used for 
growing flowers, trees, shrubs, and other plants. The act of caring for a garden by 
watering the flowers and plants and removing the weeds is called gardening."""

amazon_domains = [a_d0, a_d1, a_d2, a_d3, a_d4, a_d5, a_d6, a_d7, a_d8, a_d9, 
                  a_d10, a_d11, a_d12, a_d13, a_d14, a_d15, a_d16, a_d17, a_d18, a_d19,
                  a_d20, a_d21]


# SYNTH3 INFORMATION ----------------------------------------------------------

"""
synth3_domain = Movies, also known as films, are a type of visual communication which uses moving 
pictures and sound to tell stories or teach people something. People in every part of the world watch 
movies as a type of entertainment, a way to have fun. For some people, fun movies can mean 
movies that make them laugh, while for others it can mean movies that make them cry, or feel 
afraid. Most movies are made so that they can be shown on big screens at movie theatres and at home. 
After movies are shown on movie screens for a period of weeks or months, they may be marketed 
through several other media. They are shown on pay television or cable television, and sold or 
rented on DVD disks or videocassette tapes, so that people can watch the movies at home. You 
can also download or stream movies. Older movies are shown on television broadcasting stations.
A movie camera or video camera takes pictures very quickly, usually at 24 or 25 pictures (frames) 
every second. When a movie projector, a computer, or a television shows the pictures at that rate, 
it looks like the things shown in the set of pictures are really moving. Sound is either recorded 
at the same time, or added later. The sounds in a movie usually include the sounds of people 
talking (which is called dialogue), music (which is called the "soundtrack"), and sound effects, 
the sounds of activities that are happening in the movie (such as doors opening or guns being 
fired). In the 20th century the camera used photographic film. The product is still often called 
a "film" even though there usually is no film.
"""

synth3_domain = """good cool amazing love beautiful funny touching best bad boring terrible trash 
horrible dull obnoxious worst"""


# REDDIT INFORMATION ----------------------------------------------------------

domain_names = ["climate", "politics", "democrats", "Liberal", "progressive",
                "Feminism", "lgbt", "LGBTeens", "TwoXChromosomes", "TrollXChromosomes",
                "transgender", "atheism", "skeptic", "Freethought", "ukpolitics",
                "Libertarian", "MensRights", "climateskeptics", "conspiracy", "Republican",
                "Conservative", "4chan", "KotakuInAction", "religion", "Christianity",
                "CringeAnarchy", "PussyPass", "pussypassdenied", "MGTOW", "sjwhate", 
                "fatpeoplehate"]

r_d0 = """A community for truthful science-based news about climate and related politics and activism """

r_d1 = """for news and discussion about U.S. politics. """

r_d2 = """The Democratic Party is fighting for a country where everyone, from every walk of life, has 
an equal chance at the American dream. This sub offers daily news updates, policy analysis, links, 
and opportunities to participate in the political process. """

r_d3 = """A community for liberals. Favoring maximum individual liberty in political and social reform. 
Democratic. """

r_d4 = """A community to share stories related to the growing Modern Political and Social Progressive 
Movement. The Modern Progressive Movement advocates change and reform through directed governmental 
action. The Modern Progressive Movement stands in opposition of conservative or reactionary ideologies. """

r_d5 = """Discuss and promote awareness of issues related to equality for women."""

r_d6 = """A safe space for GSRM (Gender, Sexual, and Romantic Minority) folk to discuss their lives, 
issues, interests, and passions. LGBT is still a popular term used to discuss gender and sexual 
minorities, but all GSRM are welcome beyond lesbian, gay, bisexual, and transgender people who consent 
to participate in a safe space."""

r_d7 = """A place where LGBT teens and LGBT allies can hang out, get advice, and share content!"""

r_d8 = """Welcome to TwoXChromosomes, a subreddit for both serious and silly content, and intended for 
women's perspectives. We are a welcoming subreddit and support the rights of all genders. Posts are 
moderated for respect, equanimity, grace, and relevance."""

r_d9 = """A subreddit for rage comics and other memes with a girly slant."""

r_d10 = """Transgender news, issues, and discussion"""

r_d11 = """Welcome to r/atheism, the web's largest atheist forum. All topics related to atheism, 
agnosticism and secular living are welcome here."""

r_d12 = """A person inclined to question or doubt accepted opinions."""

r_d13 = """Freethought is an open forum dedicated to rational, logical and scientific examination of 
culture, politics, religion, science, business and more! Freethinkers reject claims and beliefs which 
are not testable and verifiable using established scientific methods, and encourage a society that 
espouses the priority of rationality and reason over dogma, emotion and pop doctrine."""

r_d14 = """Political news and debate concerning the United Kingdom."""

r_d15 = """A place to discuss libertarianism, related topics, and share things that would be of interest 
to libertarians."""

r_d16 = """The Men's Rights subreddit is a place for those who wish to discuss men's rights and the ways 
said rights are infringed upon. r/MensRights has been here since March 19, 2008."""

r_d17 = """Questioning climate related environmentalism."""

r_d18 = """The conspiracy subreddit is a thinking ground. Above all else, we respect everyone's opinions 
and ALL religious beliefs and creeds. We hope to challenge issues which have captured the public’s 
imagination, from JFK and UFOs to 9/11. This is a forum for free thinking, not hate speech. Respect other 
views and opinions, and keep an open mind.** **Our intentions are aimed towards a fairer, more transparent 
world and a better future for everyone."""

r_d19 = """Republican is a partisan subreddit. This is a place for Republicans to discuss issues with other 
Republicans."""

r_d20 = """The place for Conservatives on Reddit."""

r_d21 = """Quantum Realm time travel is used to collect stones and bring back the dusted."""

r_d22 = """KotakuInAction is the main hub for GamerGate on Reddit and welcomes discussion of community, 
industry and media issues in gaming and broader nerd culture including science fiction and comics."""

r_d23 = """The belief in and worship of a superhuman controlling power, especially a personal God or gods. 
Christianity. """

r_d24 = """Christianity is a subreddit to discuss Christianity and aspects of Christian life. All are welcome 
to participate."""

r_d25 = """A sub dedicated to cataloguing cringy content. Not affiliated with those hat subs or whatever 
#lovewins"""

r_d26 = """The Pussy Pass is a modern phenomenon where by just owning a pussy gets you benefits."""

r_d27 = """Welcome to /r/pussypassdenied, where women are not allowed to use their gender as a 
handicap or an excuse to act like assholes. Yay equality!"""

r_d28 = """This subreddit is for men going their own way, forging their own identities and paths to 
self-defined success."""

r_d29 = """Come to rage about SJW and how they are fucking shit up in real life and on the net. More 
rules and guidelines will be added as needed."""

r_d30 = """Come to rage about, hate on, and make fun of fat people. """

reddit_domains = [r_d0, r_d1, r_d2, r_d3, r_d4, r_d5, r_d6, r_d7, r_d8, r_d9, 
                  r_d10, r_d11, r_d12, r_d13, r_d14, r_d15, r_d16, r_d17, r_d18, r_d19,
                  r_d20, r_d21, r_d22, r_d23, r_d24, r_d25, r_d26, r_d27, r_d28, r_d29,
                  r_d30]


# STOPWORDS -------------------------------------------------------------------

stopwords = ['i', 'me', 'my', 'myself', 
             'we', 'our', 'ours', 'ourselves', 
             'you', "you're", "you've", "you'll", 
             "you'd", 'your', 'yours', 'yourself', 
             'yourselves', 'he', 'him', 'his', 
             'himself', 'she', "she's", 'her', 'hers', 
             'herself', 'it', "it's", 'its', 'itself', 
             'they', 'them', 'their', 'theirs', 'themselves', 
             'what', 'which', 'who', 'whom', 'this', 'that', 
             "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
             'were', 'be', 'been', 'being', 'have', 'has', 'had', 
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 
             'until', 'while', 'of', 'at', 'by', 'for', 'with', 
             'about', 'against', 'between', 'into', 'through', 
             'during', 'before', 'after', 'above', 'below', 'to', 
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
             'under', 'again', 'further', 'then', 'once', 'here', 
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 
             'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
             'don', "don't", 'should', "should've", 'now', 'd', 'll', 
             'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
             'couldn', "couldn't", 'didn', "didn't", 'doesn', 
             "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
             'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
             "weren't", 'won', "won't", 'wouldn', "wouldn't"]

